"""Stage 4 — Attention Modeling: Runtime Attention Scorer Module.

Loads calibrated machine learning model (GBDT / Random Forest / MLP),
processes extracted multimodal features, and produces calibrated 0-100 Attention Scores.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np

from src.features.feature_schema import FEATURE_COLUMNS, FrameFeatureVector


@dataclass
class AttentionScoreResult:
    """Documented result matching Stage 4 diagram output: Attention Score (0-100)."""
    track_id: int
    timestamp_sec: float
    score: float                 # Calibrated score 0-100
    category: str                # "Attentive", "Moderate", "Inattentive", "Uncertain"
    model_confidence: float
    raw_probability: float       # P(High | X) in [0, 1]
    is_uncertain_due_to_missingness: bool
    predicted_class: str = "High" # Canonical thesis 3-class label: "High", "Moderate", "Low", "Uncertain"


class AttentionScorer:
    """Computes calibrated observable attention scores (0-100) from physical modalities."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        use_supervised_checkpoint: bool = True
    ):
        self.use_supervised_checkpoint = use_supervised_checkpoint
        self.model = None
        self.feature_columns = FEATURE_COLUMNS
        self.classes = ["High", "Moderate", "Low"]
        self.model_name = "Observable Physical Behavior Index (Fallback)"

        if checkpoint_path is None:
            # Multi-candidate path resolution
            candidates = [
                Path(__file__).resolve().parent.parent.parent / "checkpoints" / "final_attention_model.joblib",
                Path("checkpoints/final_attention_model.joblib"),
                Path("classroom_attention_research/checkpoints/final_attention_model.joblib"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    checkpoint_path = candidate
                    break

        if use_supervised_checkpoint and checkpoint_path is not None and checkpoint_path.exists():
            data = joblib.load(checkpoint_path)
            self.model = data["model"]
            self.feature_columns = data.get("feature_columns", FEATURE_COLUMNS)
            self.classes = data.get("classes", ["High", "Moderate", "Low"])
            self.model_name = "Final Calibrated Ensemble Model (Random Forest)"
            print(f"[AttentionScorer] Successfully loaded production model from {checkpoint_path}")
        else:
            print(f"[AttentionScorer] Warning: No supervised model checkpoint found. Running on heuristic physical index fallback.")

    def score(self, fv: FrameFeatureVector) -> AttentionScoreResult:
        """
        Produce a calibrated 0-100 Observable Attention Score for a student frame feature vector.
        Grounds scoring strictly in observable physical markers (head pose, gaze proxy, torso posture, motion).
        """
        # Missingness & Distance Gating
        missing_count = int(fv.missing_face + fv.missing_head_pose + fv.missing_gaze + fv.missing_torso + fv.is_distant)
        is_uncertain = missing_count >= 2 or fv.is_distant > 0.5

        # 1. Primary: Supervised Calibrated Inference
        if self.model is not None:
            x_arr = np.array([[getattr(fv, col, 0.0) for col in self.feature_columns]], dtype=np.float32)
            probs = self.model.predict_proba(x_arr)[0]
            class_map = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}
            p_high = class_map.get("High", 0.0)
            p_mod = class_map.get("Moderate", 0.0)
            p_low = class_map.get("Low", 0.0)

            calibrated_score = round(float(np.clip(100.0 * (p_high + 0.5 * p_mod), 5.0, 98.0)), 1)
            best_cls = max(class_map, key=class_map.get)
            
            if is_uncertain:
                category = "Uncertain"
                pred_cls = "Uncertain"
            elif best_cls == "High":
                category = "Attentive"
                pred_cls = "High"
            elif best_cls == "Low":
                category = "Inattentive"
                pred_cls = "Low"
            else:
                category = "Moderate"
                pred_cls = "Moderate"

            return AttentionScoreResult(
                track_id=fv.track_id,
                timestamp_sec=fv.timestamp_sec,
                score=calibrated_score,
                category=category,
                model_confidence=round(float(max(probs)), 2),
                raw_probability=round(float(p_high), 4),
                is_uncertain_due_to_missingness=is_uncertain,
                predicted_class=pred_cls
            )

        # 2. Heuristic Fallback If No Model Loaded
        # 1. Gaze Proxy Sub-Score [0, 1]
        if fv.missing_gaze > 0.5 or fv.is_distant > 0.5:
            gaze_score = 0.50
        else:
            gaze_dev = float(abs(fv.gaze_horizontal))
            gaze_score = float(np.clip(1.0 - (gaze_dev / 0.85), 0.1, 1.0))

        # 2. Head Orientation Sub-Score [0, 1]
        if fv.missing_head_pose > 0.5 or fv.is_distant > 0.5:
            head_score = 0.50
        else:
            pitch = fv.head_pitch_deg
            yaw = abs(fv.head_yaw_deg)
            # Pitch penalty if head dropped steeply (< -20 deg) or tilted excessively up (> 35 deg)
            if pitch < -25.0:
                pitch_sub = 0.20  # Head slumped downward
            elif pitch < -15.0:
                pitch_sub = 0.60
            elif -15.0 <= pitch <= 25.0:
                pitch_sub = 1.00  # Frontal/attentive zone
            else:
                pitch_sub = 0.60

            # Yaw penalty if turned away (> 35 deg)
            yaw_sub = float(np.clip(1.0 - (yaw / 65.0), 0.15, 1.0))
            head_score = 0.6 * pitch_sub + 0.4 * yaw_sub

        # 3. Torso Posture Sub-Score [0, 1]
        if fv.missing_torso > 0.5:
            posture_score = 0.50
        else:
            angle = fv.torso_angle_deg
            if angle >= 75.0:
                posture_score = 1.00  # Upright posture
            elif angle >= 60.0:
                posture_score = 0.70  # Slight lean
            else:
                posture_score = 0.30  # Heavy slump / slouch

        # 4. Movement Dynamics Sub-Score [0, 1]
        energy = fv.movement_energy
        if energy > 70.0:
            motion_score = 0.35  # Extreme erratic restlessness
        elif 5.0 <= energy <= 45.0:
            motion_score = 0.90  # Active engagement (note-taking, subtle shifts)
        else:
            motion_score = 0.75  # Quiet, steady listening

        # Weighted Observable Attention Index
        if is_uncertain:
            # Rear-row / distant / occluded: heavily weight posture and movement, dampen facial cues
            raw_index = 0.55 * posture_score + 0.35 * motion_score + 0.10 * head_score
            confidence = 0.45
        else:
            raw_index = 0.35 * head_score + 0.25 * gaze_score + 0.25 * posture_score + 0.15 * motion_score
            confidence = round(0.85 - 0.10 * missing_count, 2)

        # Calibrate to 0-100 scale
        score_100 = round(float(np.clip(raw_index * 100.0, 5.0, 98.0)), 1)

        if score_100 >= 65.0:
            category = "Attentive"
        elif score_100 <= 40.0:
            category = "Inattentive"
        else:
            category = "Moderate"

        return AttentionScoreResult(
            track_id=fv.track_id,
            timestamp_sec=fv.timestamp_sec,
            score=score_100,
            category=category,
            model_confidence=confidence,
            raw_probability=round(float(raw_index), 4),
            is_uncertain_due_to_missingness=is_uncertain
        )
