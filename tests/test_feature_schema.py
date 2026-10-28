"""Tests for Phase 4: Multimodal Feature Extraction and Schema Ordering."""

import numpy as np
import pytest

from src.features.feature_schema import FEATURE_COLUMNS, FrameFeatureVector, create_feature_vector
from src.modalities.facial_attention import ModalityExtractor, StudentModalities
from src.tracking.tracker import TrackedStudent


def test_feature_columns_contract():
    """Verify that exactly 16 feature columns are defined in the canonical schema."""
    assert len(FEATURE_COLUMNS) == 16
    assert FEATURE_COLUMNS[0] == "head_pitch_deg"
    assert FEATURE_COLUMNS[1] == "head_yaw_deg"
    assert FEATURE_COLUMNS[10] == "detection_confidence"
    assert FEATURE_COLUMNS[11] == "missing_face"
    assert FEATURE_COLUMNS[15] == "is_distant"


def test_missing_features_for_distant_student():
    """Verify that distant students have missingness flags set to 1.0."""
    extractor = ModalityExtractor(min_student_area_ratio=0.01)
    student = TrackedStudent(
        track_id=10,
        bbox=(20.0, 20.0, 45.0, 65.0),
        confidence=0.75,
        keypoints_xy=np.zeros((17, 2)),
        keypoints_conf=np.zeros(17),
        area_ratio=0.003, # Very small: distant
        is_occluded=False,
        face_bbox=None
    )
    mods = extractor.extract(student, frame_shape=(720, 1280))
    fv = create_feature_vector("sess1", "clip1", student, mods, frame_idx=5, timestamp_sec=0.5)

    arr = fv.to_feature_array()
    assert len(arr) == 16
    # Check flags: missing_face, missing_head_pose, missing_gaze, is_distant
    assert fv.is_distant == 1.0
    assert fv.missing_face == 1.0
    assert fv.missing_head_pose == 1.0
    assert fv.missing_gaze == 1.0


def test_feature_vector_array_ordering():
    """Verify that to_feature_array() matches FEATURE_COLUMNS order exactly."""
    fv = FrameFeatureVector(
        session_id="s1",
        clip_id="c1",
        track_id=1,
        frame_idx=0,
        timestamp_sec=0.0,
        head_pitch_deg=-15.0,
        head_yaw_deg=5.0,
        head_roll_deg=0.0,
        head_pose_confidence=0.8,
        gaze_horizontal=0.1,
        gaze_vertical=-0.2,
        torso_angle_deg=82.0,
        shoulder_tilt_deg=1.5,
        movement_energy=12.0,
        bbox_area_ratio=0.045,
        detection_confidence=0.92,
        missing_face=0.0,
        missing_head_pose=0.0,
        missing_gaze=0.0,
        missing_torso=0.0,
        is_distant=0.0
    )
    arr = fv.to_feature_array()
    assert arr[0] == pytest.approx(-15.0)
    assert arr[1] == pytest.approx(5.0)
    assert arr[6] == pytest.approx(82.0)
    assert arr[10] == pytest.approx(0.92, abs=1e-2)
    assert arr[15] == pytest.approx(0.0)
