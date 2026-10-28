"""End-to-End Classroom Attention Video Pipeline.

Integrates the complete 6-Stage Thesis Architecture:
1. Video Ingestion & Frame Extraction (VideoReader)
2. Student Detection & Multi-Object Tracking (YOLO11-Pose + DeepSORT)
3. Facial & Postural Modality Extraction (FaceCropper + OpenSeeFaceTracker + ModalityExtractor)
4. Evidence-Based Observable Attention Scoring (AttentionScorer)
5. Temporal Attention Drop Detection (AttentionDropDetector)
6. Pedagogical Intervention Engine (RecommendationEngine)
+ Diagnostic Video Rendering (Annotated Bounding Boxes, IDs, Scores, HUD, and Telemetry)
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import cv2
import numpy as np
import torch

from src.ingestion.video_reader import VideoReader
from src.detection.pose_detector import YOLO11PoseDetector
from src.tracking.tracker import ClassroomCensusTracker
from src.modalities.face_cropper import FaceCropper
from src.modalities.openseeface_tracker import OpenSeeFaceTracker
from src.modalities.facial_attention import ModalityExtractor
from src.features.feature_schema import create_feature_vector
from src.models.attention_scorer import AttentionScorer
from src.attention_drop.detector import AttentionDropDetector
from src.recommendations.recommendation_engine import RecommendationEngine


class ClassroomAttentionPipeline:
    """Full-featured classroom attention analysis pipeline for arbitrary video input."""

    def __init__(
        self,
        sample_fps: float = 10.0,
        drop_threshold: float = 20.0,
        drop_window_sec: float = 2.0,
        cooldown_sec: float = 3.0
    ):
        self.sample_fps = sample_fps
        self.reader = VideoReader(sample_fps=sample_fps)
        self.detector = YOLO11PoseDetector()
        self.tracker = ClassroomCensusTracker()
        self.face_cropper = FaceCropper()
        self.osf_tracker = OpenSeeFaceTracker()
        self.mod_extractor = ModalityExtractor()
        self.attention_scorer = AttentionScorer()
        self.drop_detector = AttentionDropDetector(
            window_duration_sec=drop_window_sec,
            drop_magnitude_threshold=drop_threshold,
            cooldown_sec=cooldown_sec
        )
        self.rec_engine = RecommendationEngine()

    def process_video(
        self,
        input_video_path: Path,
        output_video_path: Path,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict:
        """
        Process input video end-to-end, render annotated output video,
        and return full research-grade telemetry and recommendations.
        """
        t0 = time.time()
        input_video_path = Path(input_video_path)
        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_video_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        # Reset trackers
        self.tracker.reset()
        self.osf_tracker.reset()
        self.drop_detector.reset()

        meta = self.reader.inspect(input_video_path)
        if not meta.is_valid:
            err_msg = "; ".join(meta.validation_errors) if meta.validation_errors else "Corrupt or unreadable video file."
            raise ValueError(f"Input video '{input_video_path.name}' is invalid or corrupt: {err_msg}")

        total_stream_frames = int(meta.duration_sec * self.sample_fps)
        if max_frames:
            total_stream_frames = min(total_stream_frames, max_frames)

        if progress_callback:
            progress_callback(0.05, f"Inspected video: {meta.width}x{meta.height} @ {meta.fps:.1f} FPS, {meta.duration_sec:.1f}s")

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_w, out_h = 1280, 720
        writer = cv2.VideoWriter(str(output_video_path), fourcc, self.sample_fps, (out_w, out_h))

        student_trajectories: Dict[int, List[Dict]] = {}
        all_drop_events = []
        frame_stats = []
        frame_count = 0

        stream = self.reader.stream_frames(input_video_path)

        for ts_frame in stream:
            if max_frames and frame_count >= max_frames:
                break

            frame_count += 1
            frame = ts_frame.frame.copy()
            h, w = frame.shape[:2]

            # 1. Detection
            dets = self.detector.detect(frame)

            # 2. Tracking
            tracked_students = self.tracker.update(frame, dets)

            current_frame_scores = []
            current_frame_attentive = 0
            current_frame_inattentive = 0
            current_frame_uncertain = 0

            # 3. Modalities & Attention Estimation
            for student in tracked_students:
                crop_res = self.face_cropper.crop(frame, student.bbox, student.keypoints_xy, student.keypoints_conf)
                osf_res = self.osf_tracker.process_face(student.track_id, crop_res, student.keypoints_xy, student.keypoints_conf)
                mods = self.mod_extractor.extract(student, frame.shape)

                # Feature vector
                fv = create_feature_vector(
                    session_id="runtime",
                    clip_id=input_video_path.stem,
                    student=student,
                    modalities=mods,
                    frame_idx=ts_frame.frame_idx,
                    timestamp_sec=ts_frame.timestamp_sec,
                    target_label=1
                )

                # Score
                score_res = self.attention_scorer.score(fv)
                current_frame_scores.append(score_res.score)

                if score_res.is_uncertain_due_to_missingness or mods.is_distant_student:
                    current_frame_uncertain += 1
                    status_cat = "Uncertain"
                elif score_res.score >= 60.0:
                    current_frame_attentive += 1
                    status_cat = "Attentive"
                else:
                    current_frame_inattentive += 1
                    status_cat = "Inattentive"

                # Check Drop Detector
                drop_ev = self.drop_detector.add_observation(
                    track_id=student.track_id,
                    frame_idx=ts_frame.frame_idx,
                    timestamp_sec=ts_frame.timestamp_sec,
                    score=score_res.score,
                    head_pitch_deg=osf_res.head_pitch,
                    head_yaw_deg=osf_res.head_yaw,
                    torso_angle_deg=mods.torso_angle_deg,
                    movement_energy=mods.movement_energy,
                    is_missing=not osf_res.is_valid
                )
                if drop_ev:
                    rec = self.rec_engine.generate_recommendation(drop_ev)
                    drop_data = {
                        "track_id": drop_ev.track_id,
                        "timestamp_sec": round(ts_frame.timestamp_sec, 2),
                        "frame_idx": ts_frame.frame_idx,
                        "drop_magnitude": round(drop_ev.drop_magnitude, 1),
                        "primary_modality": drop_ev.evidence,
                        "recommendation": {
                            "urgency": rec.urgency_level,
                            "explanation": rec.explanation,
                            "teaching_recommendation": rec.teaching_recommendation,
                            "intervention_suggestion": rec.intervention_suggestion
                        }
                    }
                    all_drop_events.append(drop_data)

                # Record student trajectory
                if student.track_id not in student_trajectories:
                    student_trajectories[student.track_id] = []

                behavior_tag = "Board"
                if mods.posture_valid and mods.torso_angle_deg < 55.0:
                    behavior_tag = "Slumped"
                elif osf_res.head_pitch < -20.0:
                    behavior_tag = "Reading/Desk"
                elif abs(osf_res.head_yaw) > 35.0:
                    behavior_tag = "Turned Away"
                elif mods.movement_energy > 50.0:
                    behavior_tag = "Restless"

                student_trajectories[student.track_id].append({
                    "timestamp_sec": round(ts_frame.timestamp_sec, 2),
                    "frame_idx": ts_frame.frame_idx,
                    "score": round(score_res.score, 1),
                    "category": status_cat,
                    "behavior": behavior_tag,
                    "head_pitch": round(osf_res.head_pitch, 1),
                    "head_yaw": round(osf_res.head_yaw, 1),
                    "bbox": [round(float(v), 1) for v in student.bbox]
                })

                # Visual Rendering on Frame
                x1, y1, x2, y2 = [int(v) for v in student.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                if status_cat == "Attentive":
                    bbox_color = (0, 220, 100) # Emerald
                elif status_cat == "Inattentive":
                    bbox_color = (60, 60, 240) # Red
                else:
                    bbox_color = (0, 180, 240) # Amber

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

                # Header badge (indicates whether live pose or memory-coasting reference)
                mem_tag = " [Mem Ref]" if student.tracking_mode == "COASTING" else f" [{status_cat[0]}]"
                badge_text = f"Student #{student.track_id} | {score_res.score:.0f}{mem_tag}"
                font_scale = 0.45
                thickness = 1
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), bbox_color, -1)
                cv2.putText(frame, badge_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                # Subtitle behavior
                sub_text = f"{behavior_tag}" if student.tracking_mode != "COASTING" else f"{behavior_tag} (Desk/Writing)"
                cv2.putText(frame, sub_text, (x1 + 3, min(h - 5, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, bbox_color, 1, cv2.LINE_AA)

            # Draw HUD Telemetry Header (Clean 62px bar with Census and Dynamics)
            cv2.rectangle(frame, (0, 0), (w, 62), (15, 20, 30), -1)
            mean_score = float(np.mean(current_frame_scores)) if current_frame_scores else 0.0

            hud_left = f"Smart Classroom Attention Pipeline | Time: {ts_frame.timestamp_sec:.1f}s | Frame: {frame_count}"
            hud_right = f"Census: {self.tracker.census_count} | Active: {len(tracked_students)} | Exited: {self.tracker.exited_count}"
            cv2.putText(frame, hud_left, (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 225, 235), 1, cv2.LINE_AA)
            cv2.putText(frame, hud_right, (w - 450, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 240), 2, cv2.LINE_AA)

            hud_dynamics = f"Dynamics: {self.tracker.last_event or 'STABLE'} | Class Attention: {mean_score:.1f}%"
            cv2.putText(frame, hud_dynamics, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100) if mean_score >= 60 else (60, 60, 240), 1, cv2.LINE_AA)

            hud_breakdown = f"Attentive: {current_frame_attentive}  Inattentive: {current_frame_inattentive}  Uncertain: {current_frame_uncertain}"
            cv2.putText(frame, hud_breakdown, (w - 450, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 195), 1, cv2.LINE_AA)

            # Check if recent drop event (< 1.5 sec ago) to show flashing alert
            recent_drops = [d for d in all_drop_events if abs(ts_frame.timestamp_sec - d["timestamp_sec"]) <= 1.5]
            if recent_drops:
                alert_text = f"ATTENTION DROP ALERT: Student #{recent_drops[-1]['track_id']} (-{recent_drops[-1]['drop_magnitude']}%)"
                cv2.rectangle(frame, (w // 2 - 250, 60), (w // 2 + 250, 95), (30, 30, 200), -1)
                cv2.putText(frame, alert_text, (w // 2 - 230, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

            # Resize to output
            out_frame = cv2.resize(frame, (out_w, out_h))
            writer.write(out_frame)

            frame_stats.append({
                "timestamp_sec": round(ts_frame.timestamp_sec, 2),
                "frame_idx": ts_frame.frame_idx,
                "active_students": len(tracked_students),
                "mean_attention": round(mean_score, 1),
                "attentive_count": current_frame_attentive,
                "inattentive_count": current_frame_inattentive,
                "uncertain_count": current_frame_uncertain
            })

            if progress_callback and total_stream_frames > 0:
                prog = min(0.95, 0.05 + 0.90 * (frame_count / total_stream_frames))
                progress_callback(prog, f"Processing frame {frame_count}/{total_stream_frames} ({len(tracked_students)} students tracked)...")

        writer.release()
        total_time = time.time() - t0

        # Summary statistics
        all_student_means = []
        student_summaries = []
        for tid, traj in student_trajectories.items():
            if not traj:
                continue
            scores = [t["score"] for t in traj]
            mean_s = float(np.mean(scores))
            all_student_means.append(mean_s)
            cats = [t["category"] for t in traj]
            top_cat = max(set(cats), key=cats.count)
            behaviors = [t["behavior"] for t in traj]
            top_beh = max(set(behaviors), key=behaviors.count)
            student_summaries.append({
                "track_id": tid,
                "observations": len(traj),
                "mean_score": round(mean_s, 1),
                "min_score": round(float(np.min(scores)), 1),
                "max_score": round(float(np.max(scores)), 1),
                "predominant_category": top_cat,
                "predominant_behavior": top_beh,
                "timeline": traj
            })

        overall_mean = float(np.mean(all_student_means)) if all_student_means else 0.0
        att_rate = float(np.mean([1 if s >= 60 else 0 for s in all_student_means])) if all_student_means else 0.0

        # Detailed observation counts across all student timelines
        all_obs_categories = [
            t["category"] for traj in student_trajectories.values() for t in traj
        ]
        high_obs_cnt = sum(1 for c in all_obs_categories if c in ["High", "Attentive"])
        mod_obs_cnt = sum(1 for c in all_obs_categories if c == "Moderate")
        low_obs_cnt = sum(1 for c in all_obs_categories if c in ["Low", "Inattentive"])
        unc_obs_cnt = sum(1 for c in all_obs_categories if c == "Uncertain")

        # Non-LLM Before-and-After Analysis (Phase 7)
        from src.evaluation.before_after_analysis import compare_attention_intervals
        before_after_result = compare_attention_intervals(
            student_summaries=student_summaries,
            frame_stats=frame_stats
        ).to_dict()

        if progress_callback:
            progress_callback(1.0, f"Analysis complete in {total_time:.1f}s. Annotated video rendered successfully.")

        return {
            "input_video": str(input_video_path.name),
            "output_video": str(output_video_path.name),
            "output_video_path": str(output_video_path.resolve()),
            "duration_sec": meta.duration_sec,
            "processed_frames": frame_count,
            "pipeline_fps": round(frame_count / max(0.01, total_time), 1),
            "total_processing_time_sec": round(total_time, 2),
            "census_student_count": self.tracker.census_count,
            "total_students_tracked": len(student_summaries),
            "exited_camera_count": self.tracker.exited_count,
            "vanished_students_count": self.tracker.total_vanished_students,
            "tracker_diagnostics": self.tracker.get_diagnostics(),
            "last_box_event": self.tracker.last_event or "STABLE",
            "class_average_attention": round(overall_mean, 1),
            "attentive_percentage": round(att_rate * 100, 1),
            "inattentive_percentage": round((1.0 - att_rate) * 100, 1),
            "observation_breakdown": {
                "high_count": high_obs_cnt,
                "moderate_count": mod_obs_cnt,
                "low_count": low_obs_cnt,
                "uncertain_count": unc_obs_cnt,
                "total_observations": len(all_obs_categories)
            },
            "drop_events_count": len(all_drop_events),
            "drop_events": all_drop_events,
            "before_after_assessment": before_after_result,
            "student_summaries": student_summaries,
            "frame_stats": frame_stats
        }
