"""Stage 5 — Attention Drop Detector Module.

Detects sustained attention declines with hysteresis, cooldown, and explicit
distinction between sensor occlusion/missingness and genuine behavioral drop.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
import numpy as np


@dataclass
class ScoredObservation:
    """Timestamped score record for a single student track."""
    frame_idx: int
    timestamp_sec: float
    score: float                # 0-100 calibrated score
    head_pitch_deg: float
    head_yaw_deg: float
    torso_angle_deg: float
    movement_energy: float
    is_missing: bool


@dataclass
class AttentionDropEvent:
    """Documented attention decline event for pedagogical logging and intervention."""
    event_id: str
    track_id: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    baseline_score: float
    nadir_score: float
    drop_magnitude: float
    evidence: str
    confidence: float
    is_occlusion_related: bool


class AttentionDropDetector:
    """Stateful detector monitoring student tracks for sustained attention drops."""

    def __init__(
        self,
        window_duration_sec: float = 3.0,
        drop_magnitude_threshold: float = 25.0,
        absolute_floor_threshold: float = 35.0,
        min_sustained_sec: float = 2.0,
        cooldown_sec: float = 5.0,
        min_validity_ratio: float = 0.70
    ):
        self.window_duration_sec = window_duration_sec
        self.drop_magnitude_threshold = drop_magnitude_threshold
        self.absolute_floor_threshold = absolute_floor_threshold
        self.min_sustained_sec = min_sustained_sec
        self.cooldown_sec = cooldown_sec
        self.min_validity_ratio = min_validity_ratio

        # Track history buffers: track_id -> deque of ScoredObservation
        self.buffers: Dict[int, Deque[ScoredObservation]] = {}
        # Cooldown tracker: track_id -> timestamp of last fired event
        self.last_event_time: Dict[int, float] = {}
        self.event_counter = 0

    def reset(self):
        """Reset state between video sessions."""
        self.buffers.clear()
        self.last_event_time.clear()
        self.event_counter = 0

    def add_observation(
        self,
        track_id: int,
        frame_idx: int,
        timestamp_sec: float,
        score: float,
        head_pitch_deg: float = 0.0,
        head_yaw_deg: float = 0.0,
        torso_angle_deg: float = 90.0,
        movement_energy: float = 0.0,
        is_missing: bool = False
    ) -> Optional[AttentionDropEvent]:
        """
        Ingest a new scored observation and evaluate whether an attention drop event has triggered.
        Returns AttentionDropEvent if triggered, otherwise None.
        """
        if track_id not in self.buffers:
            self.buffers[track_id] = deque()

        obs = ScoredObservation(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            score=score,
            head_pitch_deg=head_pitch_deg,
            head_yaw_deg=head_yaw_deg,
            torso_angle_deg=torso_angle_deg,
            movement_energy=movement_energy,
            is_missing=is_missing
        )
        buf = self.buffers[track_id]
        buf.append(obs)

        # Evict observations older than window_duration * 2 (to maintain pre-drop baseline)
        cutoff = timestamp_sec - (self.window_duration_sec * 2.5)
        while buf and buf[0].timestamp_sec < cutoff:
            buf.popleft()

        # Check cooldown
        if track_id in self.last_event_time:
            if (timestamp_sec - self.last_event_time[track_id]) < self.cooldown_sec:
                return None

        # Need sufficient observations spanning at least min_sustained_sec
        if not buf or (buf[-1].timestamp_sec - buf[0].timestamp_sec) < self.min_sustained_sec:
            return None

        # Split buffer into baseline (earlier half) and current evaluation window (recent half)
        eval_window_start = timestamp_sec - self.window_duration_sec
        baseline_obs = [o for o in buf if o.timestamp_sec < eval_window_start and not o.is_missing]
        recent_obs = [o for o in buf if o.timestamp_sec >= eval_window_start]

        if len(recent_obs) < 4:
            return None

        # Occlusion check: ratio of valid observations
        valid_recent = [o for o in recent_obs if not o.is_missing]
        valid_ratio = len(valid_recent) / float(len(recent_obs))

        if valid_ratio < self.min_validity_ratio:
            # High missingness: do not trigger false behavioral alarm
            return None

        # Baseline score calculation
        baseline_score = float(np.mean([o.score for o in baseline_obs])) if baseline_obs else 75.0
        recent_scores = [o.score for o in valid_recent]
        recent_mean_score = float(np.mean(recent_scores))
        nadir_score = float(np.min(recent_scores))
        drop_magnitude = baseline_score - recent_mean_score

        # Drop criteria: significant score decline OR sustained floor breach
        is_sustained_drop = (drop_magnitude >= self.drop_magnitude_threshold) or (
            recent_mean_score < self.absolute_floor_threshold
        )

        if is_sustained_drop:
            self.event_counter += 1
            self.last_event_time[track_id] = timestamp_sec

            # Evidence synthesis
            mean_pitch = float(np.mean([o.head_pitch_deg for o in valid_recent]))
            mean_yaw = float(np.mean([o.head_yaw_deg for o in valid_recent]))
            mean_torso = float(np.mean([o.torso_angle_deg for o in valid_recent]))
            mean_energy = float(np.mean([o.movement_energy for o in valid_recent]))

            evidence_items = []
            if mean_pitch < -15.0:
                evidence_items.append(f"head pitched downward ({mean_pitch:.1f} deg)")
            if abs(mean_yaw) > 30.0:
                evidence_items.append(f"head oriented away from instruction ({mean_yaw:.1f} deg)")
            if mean_torso < 70.0:
                evidence_items.append(f"slumped body posture ({mean_torso:.1f} deg)")
            if mean_energy < 5.0:
                evidence_items.append(f"low motor activity (energy: {mean_energy:.1f})")

            evidence_str = (
                f"Sustained attention score decline from baseline {baseline_score:.1f} to {recent_mean_score:.1f} "
                f"(-{drop_magnitude:.1f} pts) over {self.window_duration_sec:.1f}s. "
                f"Observable markers: {', '.join(evidence_items) if evidence_items else 'general visual disengagement'}."
            )

            # Confidence based on validity ratio and magnitude
            event_conf = float(np.clip(valid_ratio * (drop_magnitude / 50.0), 0.4, 0.95))

            return AttentionDropEvent(
                event_id=f"DROP-T{track_id:02d}-{self.event_counter:03d}",
                track_id=track_id,
                start_time_sec=round(eval_window_start, 2),
                end_time_sec=round(timestamp_sec, 2),
                duration_sec=round(timestamp_sec - eval_window_start, 2),
                baseline_score=round(baseline_score, 1),
                nadir_score=round(nadir_score, 1),
                drop_magnitude=round(drop_magnitude, 1),
                evidence=evidence_str,
                confidence=round(event_conf, 2),
                is_occlusion_related=False
            )

        return None
