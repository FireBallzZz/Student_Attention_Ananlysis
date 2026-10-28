"""Tests for Phase 7: Non-LLM Learning Improvement and Interval Comparison Module."""

import pytest
from src.evaluation.before_after_analysis import (
    compare_attention_intervals,
    BeforeAfterComparisonResult,
    IntervalMetrics,
    EPISTEMIC_DISCLAIMER
)


def make_mock_student_summaries(n_students: int = 5, pre_mean: float = 75.0, post_mean: float = 50.0):
    """Helper to generate student summaries spanning t=0.0 to t=10.0s."""
    summaries = []
    for tid in range(1, n_students + 1):
        timeline = []
        # First interval t=0.0 to 5.0s (high scores)
        for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
            timeline.append({
                "timestamp_sec": t,
                "frame_idx": int(t * 10),
                "score": pre_mean + (tid % 3) * 2.0,
                "category": "High",
                "behavior": "Board"
            })
        # Second interval t=5.5 to 10.0s (lower scores)
        for t in [5.5, 6.5, 7.5, 8.5, 9.5]:
            timeline.append({
                "timestamp_sec": t,
                "frame_idx": int(t * 10),
                "score": post_mean - (tid % 3) * 2.0,
                "category": "Low",
                "behavior": "Slumped"
            })
        summaries.append({
            "track_id": tid,
            "observations": len(timeline),
            "mean_score": (pre_mean + post_mean) / 2.0,
            "predominant_category": "Moderate",
            "predominant_behavior": "Board",
            "timeline": timeline
        })
    return summaries


def test_before_after_comparison_statistics():
    """Verify statistical comparison, Welch's t-test, Cohen's d, and effect size calculation."""
    student_summaries = make_mock_student_summaries(n_students=6, pre_mean=80.0, post_mean=45.0)
    frame_stats = [{"timestamp_sec": t, "active_students": 6} for t in range(10)]

    res = compare_attention_intervals(
        student_summaries=student_summaries,
        frame_stats=frame_stats,
        split_timestamp_sec=5.0
    )

    assert isinstance(res, BeforeAfterComparisonResult)
    assert res.has_comparison is True
    assert res.delta_mean_attention < 0.0 # dropped from 80 to 45
    assert res.t_statistic is not None
    assert res.p_value is not None
    assert res.p_value < 0.05 # Statistically significant difference
    assert res.statistically_significant is True
    assert res.cohens_d is not None
    assert abs(res.cohens_d) > 0.8 # Large effect size
    assert res.effect_size_label in ["Large", "Medium"]
    assert EPISTEMIC_DISCLAIMER in res.scientific_disclaimer


def test_before_after_proportions_and_shifts():
    """Verify category proportions (High/Moderate/Low) and observable behavioral shifts."""
    student_summaries = make_mock_student_summaries(n_students=4, pre_mean=82.0, post_mean=40.0)
    res = compare_attention_intervals(
        student_summaries=student_summaries,
        frame_stats=[],
        split_timestamp_sec=5.0
    )

    pre: IntervalMetrics = res.pre_interval
    post: IntervalMetrics = res.post_interval

    assert pre.high_percentage == 100.0
    assert pre.low_percentage == 0.0
    assert post.high_percentage == 0.0
    assert post.low_percentage == 100.0

    # Behavioral shifts: dict on dataclass, list on to_dict()
    assert res.behavioral_shifts["Board"] == -100.0
    assert res.behavioral_shifts["Slumped"] == 100.0

    dict_shifts = {item["behavior"]: item for item in res.to_dict()["behavioral_shifts"]}
    assert dict_shifts["Board"]["delta_percentage"] == -100.0
    assert dict_shifts["Slumped"]["delta_percentage"] == 100.0


def test_before_after_empty_or_single_data():
    """Verify graceful handling when timeline observations are empty."""
    res = compare_attention_intervals(
        student_summaries=[],
        frame_stats=[]
    )
    assert res.has_comparison is False
    assert res.t_statistic is None
    assert res.cohens_d is None
    assert "Insufficient observation data" in res.scientific_disclaimer
