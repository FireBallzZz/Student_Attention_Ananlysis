"""Tests for Phase 5: Attention Classification, Model Checkpoint Verification, and Inference Contracts."""

from pathlib import Path
import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.features.feature_schema import FEATURE_COLUMNS, FrameFeatureVector
from src.models.attention_scorer import AttentionScorer, AttentionScoreResult


@pytest.fixture
def checkpoint_path() -> Path:
    """Resolve the real production checkpoint file."""
    candidates = [
        Path("checkpoints/final_attention_model.joblib"),
        Path("classroom_attention_research/checkpoints/final_attention_model.joblib"),
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip("Checkpoint final_attention_model.joblib not found.")


def test_real_model_checkpoint_load_and_contracts(checkpoint_path: Path):
    """
    REAL MODEL EXECUTION: Verify the actual joblib checkpoint.
    - Model architecture: RandomForestClassifier
    - Feature schema: exactly 16 inputs matching FEATURE_COLUMNS
    - Class mapping: ['High', 'Low', 'Moderate']
    """
    ckpt = joblib.load(checkpoint_path)
    assert isinstance(ckpt, dict)
    assert "model" in ckpt
    model = ckpt["model"]
    assert isinstance(model, RandomForestClassifier)
    assert model.n_features_in_ == 16
    assert list(model.classes_) == ["High", "Low", "Moderate"]
    assert ckpt.get("feature_columns") == FEATURE_COLUMNS


def test_real_model_inference_execution(checkpoint_path: Path):
    """
    REAL MODEL EXECUTION: Verify real model inference output contracts
    on a simulated attentive student.
    """
    scorer = AttentionScorer(checkpoint_path=checkpoint_path, use_supervised_checkpoint=True)
    assert scorer.model is not None

    # Simulating attentive posture: frontal gaze, upright head, high confidence
    fv_attentive = FrameFeatureVector(
        session_id="test",
        clip_id="test",
        track_id=1,
        frame_idx=10,
        timestamp_sec=1.0,
        head_pitch_deg=2.0,
        head_yaw_deg=1.0,
        head_roll_deg=0.0,
        head_pose_confidence=0.95,
        gaze_horizontal=0.05,
        gaze_vertical=-0.02,
        torso_angle_deg=85.0,
        shoulder_tilt_deg=1.0,
        movement_energy=15.0,
        bbox_area_ratio=0.06,
        detection_confidence=0.95,
        missing_face=0.0,
        missing_head_pose=0.0,
        missing_gaze=0.0,
        missing_torso=0.0,
        is_distant=0.0
    )

    res = scorer.score(fv_attentive)
    assert isinstance(res, AttentionScoreResult)
    assert res.track_id == 1
    assert 0.0 <= res.score <= 100.0
    assert 0.0 <= res.model_confidence <= 1.0
    assert res.predicted_class in ["High", "Moderate", "Low"]
    assert res.category in ["Attentive", "Moderate", "Inattentive"]


def test_real_model_inference_inattentive(checkpoint_path: Path):
    """
    REAL MODEL EXECUTION: Verify real model inference on an inattentive profile
    (head down pitch -30, turned yaw 45, slumped torso 45).
    """
    scorer = AttentionScorer(checkpoint_path=checkpoint_path, use_supervised_checkpoint=True)

    fv_inattentive = FrameFeatureVector(
        session_id="test",
        clip_id="test",
        track_id=2,
        frame_idx=20,
        timestamp_sec=2.0,
        head_pitch_deg=-30.0,
        head_yaw_deg=45.0,
        head_roll_deg=10.0,
        head_pose_confidence=0.75,
        gaze_horizontal=0.6,
        gaze_vertical=-0.8,
        torso_angle_deg=48.0,
        shoulder_tilt_deg=8.0,
        movement_energy=4.0,
        bbox_area_ratio=0.05,
        detection_confidence=0.88,
        missing_face=0.0,
        missing_head_pose=0.0,
        missing_gaze=0.0,
        missing_torso=0.0,
        is_distant=0.0
    )

    res = scorer.score(fv_inattentive)
    assert res.score < 60.0
    assert res.predicted_class in ["Low", "Moderate"]


def test_missing_features_uncertain_classification(checkpoint_path: Path):
    """Verify that multiple missing modalities result in Uncertain classification."""
    scorer = AttentionScorer(checkpoint_path=checkpoint_path, use_supervised_checkpoint=True)

    fv_distant = FrameFeatureVector(
        session_id="test",
        clip_id="test",
        track_id=3,
        frame_idx=30,
        timestamp_sec=3.0,
        head_pitch_deg=0.0,
        head_yaw_deg=0.0,
        head_roll_deg=0.0,
        head_pose_confidence=0.0,
        gaze_horizontal=0.0,
        gaze_vertical=0.0,
        torso_angle_deg=90.0,
        shoulder_tilt_deg=0.0,
        movement_energy=0.0,
        bbox_area_ratio=0.002,
        detection_confidence=0.5,
        missing_face=1.0,
        missing_head_pose=1.0,
        missing_gaze=1.0,
        missing_torso=1.0,
        is_distant=1.0
    )

    res = scorer.score(fv_distant)
    assert res.category == "Uncertain"
    assert res.predicted_class == "Uncertain"
    assert res.is_uncertain_due_to_missingness is True


def test_fallback_heuristic_when_model_disabled():
    """MOCKED / FALLBACK TEST: Verify heuristic scoring when checkpoint is unavailable."""
    scorer = AttentionScorer(checkpoint_path=None, use_supervised_checkpoint=False)
    assert scorer.model is None

    fv = FrameFeatureVector(
        session_id="test",
        clip_id="test",
        track_id=4,
        frame_idx=40,
        timestamp_sec=4.0,
        head_pitch_deg=-5.0,
        head_yaw_deg=2.0,
        head_roll_deg=0.0,
        head_pose_confidence=0.9,
        gaze_horizontal=0.0,
        gaze_vertical=0.0,
        torso_angle_deg=88.0,
        shoulder_tilt_deg=0.0,
        movement_energy=10.0,
        bbox_area_ratio=0.04,
        detection_confidence=0.9,
        missing_face=0.0,
        missing_head_pose=0.0,
        missing_gaze=0.0,
        missing_torso=0.0,
        is_distant=0.0
    )

    res = scorer.score(fv)
    assert isinstance(res, AttentionScoreResult)
    assert res.score >= 60.0
    assert res.category == "Attentive"
