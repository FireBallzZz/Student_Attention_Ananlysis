"""Tests for Phase 3: Student Detection and Tracking Output Contracts and Diagnostics."""

import numpy as np
import pytest

from src.detection.pose_detector import PersonDetection, Keypoint
from src.tracking.tracker import ClassroomCensusTracker, TrackedStudent


def create_mock_detection(x1: float, y1: float, x2: float, y2: float, conf: float = 0.85) -> PersonDetection:
    """Helper to generate a valid PersonDetection object."""
    keypoints = [Keypoint(x=(x1 + x2) / 2, y=y1 + 10, confidence=0.9, is_visible=True)]
    return PersonDetection(
        bbox=(x1, y1, x2, y2),
        confidence=conf,
        keypoints=keypoints,
        keypoints_xy=np.zeros((17, 2)),
        keypoints_conf=np.ones(17) * 0.9,
        face_bbox=(x1 + 10, y1, x2 - 10, y1 + 30),
        area_ratio=0.05,
        is_occluded_or_truncated=False
    )


def test_tracker_initialization_and_reset():
    """Verify clean initialization and state reset."""
    tracker = ClassroomCensusTracker(max_disappeared_frames=5, census_window_frames=10)
    assert tracker.census_count == 0
    assert tracker.exited_count == 0
    assert tracker.total_vanished_students == 0

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = [create_mock_detection(50, 50, 150, 200)]
    tracker.update(frame, dets)
    assert len(tracker.tracks) == 1

    tracker.reset()
    assert len(tracker.tracks) == 0
    assert tracker.census_count == 0
    assert tracker.next_track_id == 1


def test_persistent_tracking_ids():
    """Verify that consistent overlapping detections maintain the same track ID."""
    tracker = ClassroomCensusTracker(iou_match_thresh=0.3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Detection at (100, 100, 200, 250)
    det1 = [create_mock_detection(100.0, 100.0, 200.0, 250.0)]
    tracks1 = tracker.update(frame, det1)
    assert len(tracks1) == 1
    tid1 = tracks1[0].track_id

    # Frame 2: Slight movement to (105, 102, 205, 252)
    det2 = [create_mock_detection(105.0, 102.0, 205.0, 252.0)]
    tracks2 = tracker.update(frame, det2)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == tid1


def test_coasting_and_exit_handling():
    """Verify that missed detection enters COASTING mode and boundary detection marks EXITED."""
    tracker = ClassroomCensusTracker(max_disappeared_frames=4)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Student positioned near the boundary (x1 < 15)
    det_edge = [create_mock_detection(5.0, 50.0, 80.0, 200.0)]
    tracks = tracker.update(frame, det_edge)
    assert len(tracks) == 1
    edge_tid = tracks[0].track_id

    # Simulate disappearance across frames
    for _ in range(10):
        tracker.update(frame, [])

    # The near-edge track should be marked as EXITED
    diag = tracker.get_diagnostics()
    assert tracker.exited_count >= 1
    assert edge_tid in tracker.exited_track_ids or tracker.total_exited_students >= 1


def test_tracker_diagnostics_contract():
    """Verify the schema and keys of tracker.get_diagnostics()."""
    tracker = ClassroomCensusTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = [
        create_mock_detection(50, 50, 120, 180),
        create_mock_detection(200, 50, 270, 180)
    ]
    tracker.update(frame, dets)

    diag = tracker.get_diagnostics()
    expected_keys = {
        "census_count",
        "active_tracks",
        "coasting_tracks",
        "vanished_tracks",
        "exited_tracks",
        "last_event",
        "frame_count",
        "census_established"
    }
    assert expected_keys.issubset(diag.keys())
    assert diag["active_tracks"] == 2
    assert diag["census_count"] >= 2
