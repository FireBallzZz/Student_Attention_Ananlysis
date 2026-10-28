"""Tests for Phase 8: FastAPI Backend Endpoints and Validation."""

import io
from fastapi.testclient import TestClient
import pytest

from web_app.server import app, active_jobs


@pytest.fixture
def client():
    return TestClient(app)


def test_serve_index(client: TestClient):
    """Verify that GET / returns the frontend HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Smart Classroom Attention Analytics" in response.text


def test_get_sample_videos(client: TestClient):
    """Verify that sample videos list contains clip_4 and clip_11."""
    response = client.get("/api/sample_videos")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    sample_ids = [s["id"] for s in data["samples"]]
    assert "clip_4" in sample_ids
    assert "clip_11" in sample_ids


def test_upload_empty_file_rejected(client: TestClient):
    """Verify that uploading a 0-byte video is rejected with 400 Bad Request."""
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/api/analyze",
        files={"video_file": ("empty.mp4", empty_file, "video/mp4")},
        data={"max_duration_sec": 5.0}
    )
    assert response.status_code == 400
    assert "0 bytes" in response.json()["detail"]


def test_upload_invalid_extension_rejected(client: TestClient):
    """Verify that non-video file extensions are rejected with 400 Bad Request."""
    text_file = io.BytesIO(b"just text")
    response = client.post(
        "/api/analyze",
        files={"video_file": ("test.txt", text_file, "text/plain")},
        data={"max_duration_sec": 5.0}
    )
    assert response.status_code == 400
    assert "Invalid video file format" in response.json()["detail"]


def test_get_analysis_not_found(client: TestClient):
    """Verify that querying a non-existent job ID returns 404."""
    response = client.get("/api/analysis/nonexistent_job_123")
    assert response.status_code == 404


def test_compare_intervals_endpoint(client: TestClient):
    """Verify that /api/compare_intervals computes statistics on active jobs."""
    test_job_id = "test_job_endpoint_1"
    active_jobs[test_job_id] = {
        "job_id": test_job_id,
        "student_summaries": [
            {
                "track_id": 1,
                "timeline": [
                    {"timestamp_sec": 1.0, "score": 80.0, "category": "High", "behavior": "Board"},
                    {"timestamp_sec": 2.0, "score": 85.0, "category": "High", "behavior": "Board"},
                    {"timestamp_sec": 6.0, "score": 40.0, "category": "Low", "behavior": "Slumped"},
                    {"timestamp_sec": 7.0, "score": 45.0, "category": "Low", "behavior": "Slumped"},
                ]
            }
        ],
        "frame_stats": []
    }

    response = client.post(
        "/api/compare_intervals",
        json={"job_id": test_job_id, "split_timestamp_sec": 3.5}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["has_comparison"] is True
    assert result["delta_mean_attention"] == -40.0
    assert "SCIENTIFIC INTEGRITY NOTICE" in result["scientific_disclaimer"]
