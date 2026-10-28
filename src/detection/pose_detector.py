"""Stage 2 — Detection and Tracking: Pose Estimation Module.

Uses YOLO11-Pose to detect students, estimate 17 body keypoints,
and compute occlusion/visibility flags.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Keypoint:
    """Single 2D body keypoint with coordinate and confidence."""
    x: float
    y: float
    confidence: float
    is_visible: bool


@dataclass
class PersonDetection:
    """Structured container for single detected student in a frame."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    confidence: float
    keypoints: List[Keypoint]
    keypoints_xy: np.ndarray                 # shape: (17, 2)
    keypoints_conf: np.ndarray               # shape: (17,)
    area_ratio: float
    is_occluded_or_truncated: bool
    face_bbox: Optional[Tuple[float, float, float, float]] = None


# COCO Keypoint Index Map
KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}


class YOLO11PoseDetector:
    """YOLO11-Pose student detector and keypoint extractor."""

    def __init__(
        self,
        model_name: str = "classroom_attention_research/checkpoints/yolo11n-pose.pt",
        conf_threshold: float = 0.18,
        iou_threshold: float = 0.50,
        imgsz: int = 1280,
        device: Optional[str] = None
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        print(f"[YOLO11-Pose] Initializing model '{model_name}' on device '{self.device}' (conf={conf_threshold}, imgsz={imgsz})...")
        self.model = YOLO(model_name)
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Run inference on frame and return list of PersonDetection objects.
        """
        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_h * frame_w)

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        keypoints_data = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None

        # Spatial de-duplication (suppress duplicate overlapping boxes)
        sort_indices = np.argsort(scores)[::-1]
        keep_indices = []
        for i in sort_indices:
            box_i = boxes[i]
            is_dup = False
            for k in keep_indices:
                box_k = boxes[k]
                xA = max(box_i[0], box_k[0])
                yA = max(box_i[1], box_k[1])
                xB = min(box_i[2], box_k[2])
                yB = min(box_i[3], box_k[3])
                inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                area_k = (box_k[2] - box_k[0]) * (box_k[3] - box_k[1])
                denom = float(area_i + area_k - inter)
                iou = inter / denom if denom > 0 else 0.0
                if iou > self.iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep_indices.append(i)

        for idx in keep_indices:
            box = boxes[idx]
            score = scores[idx]
            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            box_area = w * h
            area_ratio = box_area / frame_area

            # Edge truncation check (within 2% of frame boundary)
            near_left = x1 < 0.02 * frame_w
            near_top = y1 < 0.02 * frame_h
            near_right = x2 > 0.98 * frame_w
            near_bottom = y2 > 0.98 * frame_h
            is_edge_truncated = near_left or near_top or near_right or near_bottom

            kpts_list = []
            kpts_xy = np.zeros((17, 2), dtype=np.float32)
            kpts_conf = np.zeros(17, dtype=np.float32)

            if keypoints_data is not None and idx < len(keypoints_data):
                raw_kpts = keypoints_data[idx]  # shape: (17, 3) -> [x, y, conf]
                for k_i in range(min(17, len(raw_kpts))):
                    kx, ky, kc = float(raw_kpts[k_i][0]), float(raw_kpts[k_i][1]), float(raw_kpts[k_i][2])
                    kpts_xy[k_i] = [kx, ky]
                    kpts_conf[k_i] = kc
                    kpts_list.append(Keypoint(x=kx, y=ky, confidence=kc, is_visible=kc > 0.3))

            # Derive face bounding box from head keypoints (nose, eyes, ears: indices 0..4)
            face_kpts = [kpts_xy[i] for i in range(5) if kpts_conf[i] > 0.35]
            face_bbox = None
            if len(face_kpts) >= 2:
                fxs = [p[0] for p in face_kpts]
                fys = [p[1] for p in face_kpts]
                min_fx, max_fx = min(fxs), max(fxs)
                min_fy, max_fy = min(fys), max(fys)
                # Expand box slightly to cover chin and forehead
                f_w = max(10.0, max_fx - min_fx)
                f_h = max(10.0, max_fy - min_fy)
                face_x1 = max(0.0, min_fx - 0.25 * f_w)
                face_y1 = max(0.0, min_fy - 0.4 * f_h)
                face_x2 = min(float(frame_w), max_fx + 0.25 * f_w)
                face_y2 = min(float(frame_h), max_fy + 0.3 * f_h)
                face_bbox = (face_x1, face_y1, face_x2, face_y2)

            # Check occlusion based on missing torso keypoints (shoulders/hips)
            torso_conf = [kpts_conf[i] for i in [5, 6, 11, 12]]
            is_occluded = is_edge_truncated or (np.mean(torso_conf) < 0.3)

            detections.append(PersonDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(score),
                keypoints=kpts_list,
                keypoints_xy=kpts_xy,
                keypoints_conf=kpts_conf,
                area_ratio=area_ratio,
                is_occluded_or_truncated=is_occluded,
                face_bbox=face_bbox
            ))

        return detections
