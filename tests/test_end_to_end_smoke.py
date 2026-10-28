"""Tests for Phase 9: End-to-End Pipeline Smoke Test with Small Authorized Fixture."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.pipeline.video_pipeline import ClassroomAttentionPipeline


@pytest.fixture
def small_video_fixture(tmp_path: Path) -> Path:
    """Create a temporary 10-frame video fixture."""
    vid_path = tmp_path / "smoke_input.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (640, 480))
    for i in range(10):
        # Draw a synthetic moving rectangle representing a seated student
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        x = 200 + i * 2
        y = 150
        cv2.rectangle(frame, (x, y), (x + 80, y + 150), (200, 200, 200), -1)
        # Add a face circle
        cv2.circle(frame, (x + 40, y + 30), 20, (230, 210, 190), -1)
        writer.write(frame)
    writer.release()
    return vid_path


def test_end_to_end_pipeline_execution(small_video_fixture: Path, tmp_path: Path):
    """
    Execute full ClassroomAttentionPipeline end-to-end on fixture:
    - Input decoding
    - Student detection & tracking
    - Multimodal feature extraction
    - Attention classification
    - Diagnostic video rendering
    - Observation breakdown (High/Moderate/Low/Uncertain)
    - Before-and-after analysis calculation
    """
    output_path = tmp_path / "smoke_output.mp4"

    pipeline = ClassroomAttentionPipeline(sample_fps=10.0, drop_threshold=20.0)
    results = pipeline.process_video(
        input_video_path=small_video_fixture,
        output_video_path=output_path,
        max_frames=10
    )

    # 1. Output video verification
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    cap = cv2.VideoCapture(str(output_path))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 5
    cap.release()

    # 2. Results contract verification
    assert results["processed_frames"] >= 5
    assert "census_student_count" in results
    assert "tracker_diagnostics" in results
    assert "observation_breakdown" in results
    assert "before_after_assessment" in results
    assert "student_summaries" in results
    assert "frame_stats" in results

    # Observation breakdown must report counts
    obs = results["observation_breakdown"]
    assert "high_count" in obs
    assert "moderate_count" in obs
    assert "low_count" in obs
    assert "uncertain_count" in obs
    assert "total_observations" in obs
