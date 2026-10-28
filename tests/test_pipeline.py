"""Automated unit and integration tests for critical pipeline modules."""

import numpy as np
import pytest

from src.detection.pose_detector import PersonDetection, Keypoint
from src.tracking.tracker import TrackedStudent
from src.modalities.facial_attention import ModalityExtractor
from src.features.feature_schema import create_feature_vector, FEATURE_COLUMNS
from src.attention_drop.detector import AttentionDropDetector, AttentionDropEvent
from src.recommendations.recommendation_engine import RecommendationEngine


def test_modality_extractor_distant_gating():
    """Verify that distant students have facial/gaze modalities explicitly flagged as missing."""
    extractor = ModalityExtractor(min_student_area_ratio=0.01)

    # Mock student with tiny area ratio (0.002)
    student = TrackedStudent(
        track_id=1,
        bbox=(10.0, 10.0, 40.0, 50.0),
        confidence=0.8,
        keypoints_xy=np.zeros((17, 2)),
        keypoints_conf=np.zeros(17),
        area_ratio=0.002,
        is_occluded=False,
        face_bbox=(15.0, 15.0, 25.0, 25.0)
    )

    mods = extractor.extract(student, frame_shape=(720, 1280))
    assert mods.is_distant_student is True
    assert mods.head_pose_valid is False
    assert mods.gaze_valid is False
    assert mods.modality_missing_flags["face_landmarks"] is True
    assert mods.modality_missing_flags["gaze"] is True


def test_feature_vector_schema():
    """Verify feature vector serialization and presence of all schema columns."""
    extractor = ModalityExtractor()
    student = TrackedStudent(
        track_id=2,
        bbox=(100.0, 100.0, 250.0, 400.0),
        confidence=0.9,
        keypoints_xy=np.zeros((17, 2)),
        keypoints_conf=np.ones(17) * 0.8,
        area_ratio=0.04,
        is_occluded=False,
        face_bbox=(120.0, 100.0, 180.0, 160.0)
    )
    mods = extractor.extract(student, frame_shape=(720, 1280))

    fv = create_feature_vector(
        session_id="test_session",
        clip_id="test_clip",
        student=student,
        modalities=mods,
        frame_idx=10,
        timestamp_sec=2.0,
        target_label=1
    )
    arr = fv.to_feature_array()
    assert len(arr) == len(FEATURE_COLUMNS)
    assert fv.target_label == 1
    assert fv.track_id == 2


def test_attention_drop_detector_hysteresis_and_cooldown():
    """Verify that drop detector triggers on sustained decline and respects cooldown."""
    detector = AttentionDropDetector(
        window_duration_sec=2.0,
        drop_magnitude_threshold=20.0,
        min_sustained_sec=1.5,
        cooldown_sec=4.0
    )

    # Ingest baseline high scores (80.0) from t=0.0 to t=2.0
    for t in np.linspace(0.0, 2.0, 10):
        detector.add_observation(track_id=5, frame_idx=int(t * 30), timestamp_sec=float(t), score=80.0)

    # Ingest steep drop (30.0) from t=2.1 to t=4.0
    event = None
    for t in np.linspace(2.1, 4.0, 10):
        ev = detector.add_observation(
            track_id=5,
            frame_idx=int(t * 30),
            timestamp_sec=float(t),
            score=30.0,
            head_pitch_deg=-25.0,
            torso_angle_deg=60.0
        )
        if ev is not None:
            event = ev

    assert event is not None
    assert event.track_id == 5
    assert event.drop_magnitude >= 20.0
    assert "Observable markers:" in event.evidence

    # Immediately after drop, cooldown must suppress second alarm
    suppressed = detector.add_observation(track_id=5, frame_idx=130, timestamp_sec=4.5, score=25.0)
    assert suppressed is None


def test_recommendation_engine():
    """Verify recommendation generation from drop event."""
    engine = RecommendationEngine()
    drop_event = AttentionDropEvent(
        event_id="TEST-01",
        track_id=3,
        start_time_sec=2.0,
        end_time_sec=5.0,
        duration_sec=3.0,
        baseline_score=80.0,
        nadir_score=35.0,
        drop_magnitude=45.0,
        evidence="head pitched downward (-28.0 deg) and slumped body posture (62.0 deg)",
        confidence=0.85,
        is_occlusion_related=False
    )
    rec = engine.generate_recommendation(drop_event)
    assert rec.target_track_id == 3
    assert len(rec.explanation) > 0
    assert len(rec.teaching_recommendation) > 0
    assert len(rec.intervention_suggestion) > 0
    assert "NOTICE:" in rec.caution_disclaimer or "CAUTION:" in rec.caution_disclaimer
