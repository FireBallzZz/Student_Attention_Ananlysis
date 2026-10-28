"""Stage 4 — Attention Feature Schema and Vectorization.

Defines the exact multimodal feature schema, preserves timestamps and track IDs,
and encodes explicit missingness indicators.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.modalities.facial_attention import StudentModalities
from src.tracking.tracker import TrackedStudent


# Schema Column Definitions
FEATURE_COLUMNS = [
    # Continuous Observable Features
    "head_pitch_deg",
    "head_yaw_deg",
    "head_roll_deg",
    "head_pose_confidence",
    "gaze_horizontal",
    "gaze_vertical",
    "torso_angle_deg",
    "shoulder_tilt_deg",
    "movement_energy",
    "bbox_area_ratio",
    "detection_confidence",
    # Explicit Missingness / Context Flags (1.0 = missing, 0.0 = observed)
    "missing_face",
    "missing_head_pose",
    "missing_gaze",
    "missing_torso",
    "is_distant"
]


@dataclass
class FrameFeatureVector:
    """Standardized record of student observables at one timestamp."""
    session_id: str
    clip_id: str
    track_id: int
    frame_idx: int
    timestamp_sec: float

    # Features
    head_pitch_deg: float
    head_yaw_deg: float
    head_roll_deg: float
    head_pose_confidence: float
    gaze_horizontal: float
    gaze_vertical: float
    torso_angle_deg: float
    shoulder_tilt_deg: float
    movement_energy: float
    bbox_area_ratio: float
    detection_confidence: float

    # Missingness
    missing_face: float
    missing_head_pose: float
    missing_gaze: float
    missing_torso: float
    is_distant: float

    # Optional Ground-Truth Target Label
    target_label: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_feature_array(self) -> np.ndarray:
        return np.array([getattr(self, col) for col in FEATURE_COLUMNS], dtype=np.float32)


def create_feature_vector(
    session_id: str,
    clip_id: str,
    student: TrackedStudent,
    modalities: StudentModalities,
    frame_idx: int,
    timestamp_sec: float,
    target_label: Optional[int] = None
) -> FrameFeatureVector:
    """Builds a verified FrameFeatureVector from tracked student and modalities."""
    m_flags = modalities.modality_missing_flags

    return FrameFeatureVector(
        session_id=session_id,
        clip_id=clip_id,
        track_id=student.track_id,
        frame_idx=frame_idx,
        timestamp_sec=round(timestamp_sec, 3),
        head_pitch_deg=modalities.head_pitch_deg if modalities.head_pose_valid else 0.0,
        head_yaw_deg=modalities.head_yaw_deg if modalities.head_pose_valid else 0.0,
        head_roll_deg=modalities.head_roll_deg if modalities.head_pose_valid else 0.0,
        head_pose_confidence=modalities.head_pose_confidence,
        gaze_horizontal=modalities.gaze_horizontal if modalities.gaze_valid else 0.0,
        gaze_vertical=modalities.gaze_vertical if modalities.gaze_valid else 0.0,
        torso_angle_deg=modalities.torso_angle_deg if modalities.posture_valid else 90.0,
        shoulder_tilt_deg=modalities.shoulder_tilt_deg if modalities.posture_valid else 0.0,
        movement_energy=modalities.movement_energy,
        bbox_area_ratio=round(student.area_ratio, 4),
        detection_confidence=round(student.confidence, 3),
        missing_face=1.0 if m_flags.get("face_landmarks", True) else 0.0,
        missing_head_pose=1.0 if m_flags.get("head_pose", True) else 0.0,
        missing_gaze=1.0 if m_flags.get("gaze", True) else 0.0,
        missing_torso=1.0 if m_flags.get("torso_posture", True) else 0.0,
        is_distant=1.0 if modalities.is_distant_student else 0.0,
        target_label=target_label
    )


def feature_vectors_to_dataframe(vectors: List[FrameFeatureVector]) -> pd.DataFrame:
    """Converts a collection of FrameFeatureVector records into a clean pandas DataFrame."""
    return pd.DataFrame([v.to_dict() for v in vectors])
