"""Stage 2 — Detection and Tracking: Multi-Student Tracker Module.

Implements ByteTrack and DeepSORT wrappers to track students continuously,
assigning persistent track IDs, maintaining trajectory history,
and evaluating tracking performance metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.detection.pose_detector import PersonDetection, Keypoint


@dataclass
class TrackedStudent:
    """Represents an identified, tracked student in a single video frame."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    keypoints_xy: np.ndarray                 # shape: (17, 2)
    keypoints_conf: np.ndarray               # shape: (17,)
    area_ratio: float
    is_occluded: bool
    face_bbox: Optional[Tuple[float, float, float, float]] = None
    center: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    tracking_mode: str = "ACTIVE"


class BaseStudentTracker(ABC):
    """Abstract interface for classroom student tracking algorithms."""

    @abstractmethod
    def reset(self):
        """Reset internal track state between videos."""
        pass

    @abstractmethod
    def update(self, frame: np.ndarray, detections: List[PersonDetection]) -> List[TrackedStudent]:
        """Update tracker with frame and detections, returning tracked students with persistent IDs."""
        pass


class ByteTrackTracker(BaseStudentTracker):
    """ByteTrack wrapper leveraging Ultralytics integrated multi-object tracking."""

    def __init__(self, model_path: str = "classroom_attention_research/checkpoints/yolo11n-pose.pt", conf_thresh: float = 0.35, device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tracker_name = "ByteTrack"
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.prev_centers: Dict[int, Tuple[float, float]] = {}

    def reset(self):
        self.prev_centers.clear()

    def update(self, frame: np.ndarray, detections: List[PersonDetection]) -> List[TrackedStudent]:
        # Run YOLO with ByteTrack tracker
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )

        tracked_students = []
        if not results or len(results) == 0:
            return tracked_students

        res = results[0]
        if res.boxes is None or res.boxes.id is None:
            return tracked_students

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        track_ids = res.boxes.id.int().cpu().numpy()
        keypoints_data = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None

        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_h * frame_w)

        for idx, (box, score, t_id) in enumerate(zip(boxes, scores, track_ids)):
            x1, y1, x2, y2 = box.tolist()
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            area_ratio = (w * h) / frame_area

            # Velocity calculation
            vx, vy = 0.0, 0.0
            if t_id in self.prev_centers:
                px, py = self.prev_centers[t_id]
                vx, vy = cx - px, cy - py
            self.prev_centers[t_id] = (cx, cy)

            kpts_xy = np.zeros((17, 2), dtype=np.float32)
            kpts_conf = np.zeros(17, dtype=np.float32)

            if keypoints_data is not None and idx < len(keypoints_data):
                raw_kpts = keypoints_data[idx]
                for k_i in range(min(17, len(raw_kpts))):
                    kpts_xy[k_i] = [float(raw_kpts[k_i][0]), float(raw_kpts[k_i][1])]
                    kpts_conf[k_i] = float(raw_kpts[k_i][2])

            # Derive face bbox from head keypoints
            face_kpts = [kpts_xy[i] for i in range(5) if kpts_conf[i] > 0.35]
            face_bbox = None
            if len(face_kpts) >= 2:
                fxs = [p[0] for p in face_kpts]
                fys = [p[1] for p in face_kpts]
                fw = max(10.0, max(fxs) - min(fxs))
                fh = max(10.0, max(fys) - min(fys))
                face_bbox = (
                    max(0.0, min(fxs) - 0.25 * fw),
                    max(0.0, min(fys) - 0.4 * fh),
                    min(float(frame_w), max(fxs) + 0.25 * fw),
                    min(float(frame_h), max(fys) + 0.3 * fh)
                )

            # Torso occlusion
            torso_conf = [kpts_conf[i] for i in [5, 6, 11, 12]]
            is_occluded = (np.mean(torso_conf) < 0.3) or (x1 < 5 or y1 < 5 or x2 > frame_w - 5 or y2 > frame_h - 5)

            tracked_students.append(TrackedStudent(
                track_id=int(t_id),
                bbox=(x1, y1, x2, y2),
                confidence=float(score),
                keypoints_xy=kpts_xy,
                keypoints_conf=kpts_conf,
                area_ratio=area_ratio,
                is_occluded=is_occluded,
                face_bbox=face_bbox,
                center=(cx, cy),
                velocity=(vx, vy)
            ))

        return tracked_students


class DeepSortTracker(BaseStudentTracker):
    """DeepSORT wrapper using deep-sort-realtime with Kalman filtering and appearance embeddings."""

    def __init__(self, max_age: int = 30, n_init: int = 3, max_cosine_distance: float = 0.2):
        self.tracker_name = "DeepSORT"
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=100,
            override_track_class=None,
            embedder=None,
            half=True,
            bgr=True
        )
        self.prev_centers: Dict[int, Tuple[float, float]] = {}

    def reset(self):
        self.tracker.delete_all_tracks()
        self.prev_centers.clear()

    def update(self, frame: np.ndarray, detections: List[PersonDetection]) -> List[TrackedStudent]:
        # Format detections for DeepSort: ([left, top, w, h], confidence, detection_class)
        raw_detections = []
        embeds = []
        frame_h, frame_w = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            raw_detections.append(([x1, y1, w, h], det.confidence, "person"))

            # Appearance embedding from student bounding box crop
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(frame_w, int(x2)), min(frame_h, int(y2))
            crop = frame[iy1:iy2, ix1:ix2]
            if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
                emb = cv2.normalize(hist, hist).flatten().astype(np.float32)
            else:
                emb = np.zeros(128, dtype=np.float32)
            embeds.append(emb)

        tracks = self.tracker.update_tracks(raw_detections, embeds=embeds, frame=frame)
        tracked_students = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = int(track.track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            vx, vy = 0.0, 0.0
            if track_id in self.prev_centers:
                px, py = self.prev_centers[track_id]
                vx, vy = cx - px, cy - py
            self.prev_centers[track_id] = (cx, cy)

            # Match nearest detection to associate pose keypoints
            best_det = None
            min_dist = float("inf")
            for det in detections:
                det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                det_cy = (det.bbox[1] + det.bbox[3]) / 2.0
                dist = np.hypot(cx - det_cx, cy - det_cy)
                if dist < min_dist:
                    min_dist = dist
                    best_det = det

            if best_det is not None and min_dist < max(x2 - x1, y2 - y1):
                kpts_xy = best_det.keypoints_xy
                kpts_conf = best_det.keypoints_conf
                area_ratio = best_det.area_ratio
                is_occluded = best_det.is_occluded_or_truncated
                face_bbox = best_det.face_bbox
                confidence = best_det.confidence
            else:
                kpts_xy = np.zeros((17, 2), dtype=np.float32)
                kpts_conf = np.zeros(17, dtype=np.float32)
                area_ratio = ((x2 - x1) * (y2 - y1)) / float(frame.shape[0] * frame.shape[1])
                is_occluded = True
                face_bbox = None
                confidence = 0.5

            tracked_students.append(TrackedStudent(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                keypoints_xy=kpts_xy,
                keypoints_conf=kpts_conf,
                area_ratio=area_ratio,
                is_occluded=is_occluded,
                face_bbox=face_bbox,
                center=(cx, cy),
                velocity=(vx, vy)
            ))

        return tracked_students


class ClassroomCensusTracker(BaseStudentTracker):
    """
    Classroom Census & Temporal Memory Tracker.
    
    Addresses:
    1. Initial Student Census: Counts all seated students during initial frames.
    2. Previous Frame Reference: Sustains bounding boxes when students look down, write, or slump.
    3. Anti-Duplicacy: Enforces strict IoU de-duplication to prevent duplicate boxes on one student.
    4. Exit Detection: Smoothly handles students moving out of camera view without false boxes.
    5. Box Dynamics Telemetry: Tracks when boxes increase (new entry) or vanish (exit/occlusion).
    """

    def __init__(
        self,
        max_disappeared_frames: int = 25,
        iou_match_thresh: float = 0.25,
        iou_dedup_thresh: float = 0.45,
        census_window_frames: int = 15
    ):
        self.tracker_name = "ClassroomCensusTracker"
        self.max_disappeared = max_disappeared_frames
        self.iou_match_thresh = iou_match_thresh
        self.iou_dedup_thresh = iou_dedup_thresh
        self.census_window_frames = census_window_frames

        self.next_track_id = 1
        self.tracks: Dict[int, Dict] = {}
        self.frame_count = 0
        self.census_established = False
        self.initial_census_count = 0
        self.prev_active_count = 0
        self.last_event: Optional[str] = None
        self.exited_track_ids = set()
        self.vanished_track_ids = set()
        self.total_exited_students = 0
        self.total_vanished_students = 0

    def reset(self):
        self.next_track_id = 1
        self.tracks.clear()
        self.frame_count = 0
        self.census_established = False
        self.initial_census_count = 0
        self.prev_active_count = 0
        self.last_event = None
        self.exited_track_ids.clear()
        self.vanished_track_ids.clear()
        self.total_exited_students = 0
        self.total_vanished_students = 0

    @property
    def census_count(self) -> int:
        return max(self.initial_census_count, len([t for t in self.tracks.values() if t["status"] in ["ACTIVE", "COASTING"]]))

    @property
    def active_count(self) -> int:
        return len([t for t in self.tracks.values() if t["status"] in ["ACTIVE", "COASTING"]])

    @property
    def exited_count(self) -> int:
        return self.total_exited_students

    @staticmethod
    def _compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    def update(self, frame: np.ndarray, detections: List[PersonDetection]) -> List[TrackedStudent]:
        self.frame_count += 1
        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_h * frame_w)

        # 1. De-duplicate incoming detections to avoid duplicate boxes
        valid_dets = []
        for det in detections:
            is_dup = False
            for existing in valid_dets:
                if self._compute_iou(det.bbox, existing.bbox) > self.iou_dedup_thresh:
                    is_dup = True
                    break
            if not is_dup:
                valid_dets.append(det)

        matched_track_ids = set()
        unmatched_dets = list(range(len(valid_dets)))

        # 2. Match incoming detections to active tracks
        if len(self.tracks) > 0 and len(valid_dets) > 0:
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(valid_dets)), dtype=np.float32)

            for i, tid in enumerate(track_ids):
                for j, det in enumerate(valid_dets):
                    iou_matrix[i, j] = self._compute_iou(self.tracks[tid]["bbox"], det.bbox)

            while True:
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_match_thresh:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]

                det = valid_dets[j]
                # Smooth bounding box with previous frame (EMA smoothing)
                old_box = self.tracks[tid]["bbox"]
                smooth_box = (
                    0.7 * det.bbox[0] + 0.3 * old_box[0],
                    0.7 * det.bbox[1] + 0.3 * old_box[1],
                    0.7 * det.bbox[2] + 0.3 * old_box[2],
                    0.7 * det.bbox[3] + 0.3 * old_box[3],
                )

                self.tracks[tid]["bbox"] = smooth_box
                self.tracks[tid]["confidence"] = det.confidence
                self.tracks[tid]["keypoints_xy"] = det.keypoints_xy
                self.tracks[tid]["keypoints_conf"] = det.keypoints_conf
                self.tracks[tid]["area_ratio"] = det.area_ratio
                self.tracks[tid]["is_occluded"] = det.is_occluded_or_truncated
                self.tracks[tid]["face_bbox"] = det.face_bbox
                self.tracks[tid]["disappeared"] = 0
                self.tracks[tid]["status"] = "ACTIVE"
                self.tracks[tid]["last_seen_frame"] = self.frame_count

                matched_track_ids.add(tid)
                if j in unmatched_dets:
                    unmatched_dets.remove(j)

                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # 3. Handle unmatched tracks (Reference previous frame)
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self.tracks[tid]["disappeared"] += 1
                box = self.tracks[tid]["bbox"]

                # Check if student moved out of camera bounds
                near_edge = (box[0] < 15 or box[1] < 15 or box[2] > frame_w - 15 or box[3] > frame_h - 15)

                if near_edge and self.tracks[tid]["disappeared"] > 8:
                    self.tracks[tid]["status"] = "EXITED"
                    if tid not in self.exited_track_ids:
                        self.exited_track_ids.add(tid)
                        self.total_exited_students += 1
                        self.last_event = f"STUDENT_EXITED_FRAME (Track #{tid})"
                elif self.tracks[tid]["disappeared"] > self.max_disappeared:
                    self.tracks[tid]["status"] = "VANISHED"
                    if tid not in self.vanished_track_ids:
                        self.vanished_track_ids.add(tid)
                        self.total_vanished_students += 1
                        self.last_event = f"TRACK_VANISHED (Track #{tid})"
                else:
                    # Keep track active using previous frame reference (coasting)
                    self.tracks[tid]["status"] = "COASTING"
                    self.tracks[tid]["is_occluded"] = True

        # Purge vanished tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["status"] in ["VANISHED", "EXITED"] and self.tracks[tid]["disappeared"] > self.max_disappeared:
                del self.tracks[tid]

        # 4. Register new tracks from unmatched detections (avoid duplicacy)
        for j in unmatched_dets:
            det = valid_dets[j]
            is_dup = False
            for tid, tdata in self.tracks.items():
                if tdata["status"] in ["ACTIVE", "COASTING"]:
                    if self._compute_iou(tdata["bbox"], det.bbox) > self.iou_dedup_thresh:
                        is_dup = True
                        break
            if not is_dup:
                self.tracks[self.next_track_id] = {
                    "track_id": self.next_track_id,
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "keypoints_xy": det.keypoints_xy,
                    "keypoints_conf": det.keypoints_conf,
                    "area_ratio": det.area_ratio,
                    "is_occluded": det.is_occluded_or_truncated,
                    "face_bbox": det.face_bbox,
                    "disappeared": 0,
                    "status": "ACTIVE",
                    "first_seen_frame": self.frame_count,
                    "last_seen_frame": self.frame_count
                }
                self.next_track_id += 1

        # 5. Census Counting & Track Increase / Vanish Diagnostics
        active_visible = [t for t in self.tracks.values() if t["status"] in ["ACTIVE", "COASTING"]]
        active_count = len(active_visible)

        if not self.census_established:
            if self.frame_count <= self.census_window_frames:
                self.initial_census_count = max(self.initial_census_count, active_count)
            else:
                self.census_established = True

        if self.frame_count > 1:
            if active_count > self.prev_active_count:
                self.last_event = f"BOX_INCREASED (+{active_count - self.prev_active_count})"
            elif active_count < self.prev_active_count:
                self.last_event = f"BOX_VANISHED (-{self.prev_active_count - active_count})"
            else:
                self.last_event = "STABLE"

        self.prev_active_count = active_count

        # Convert to standard TrackedStudent objects
        results_tracked = []
        for t in active_visible:
            x1, y1, x2, y2 = t["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            results_tracked.append(TrackedStudent(
                track_id=t["track_id"],
                bbox=t["bbox"],
                confidence=t["confidence"],
                keypoints_xy=t["keypoints_xy"],
                keypoints_conf=t["keypoints_conf"],
                area_ratio=t["area_ratio"],
                is_occluded=t["is_occluded"],
                face_bbox=t["face_bbox"],
                center=(cx, cy),
                velocity=(0.0, 0.0),
                tracking_mode=t["status"]
            ))

        return results_tracked

