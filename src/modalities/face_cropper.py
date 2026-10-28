"""Stage 3 — Facial-Attention Modalities: Face Cropping Module.

Extracts student face crops from person bounding boxes and upper-body keypoints.
Handles image boundaries, scale normalization, and resolution/quality gating.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class FaceCropResult:
    """Result of face extraction for a student detection."""
    face_crop: Optional[np.ndarray]
    bbox_px: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    face_width: int
    face_height: int
    is_valid_resolution: bool
    quality_score: float  # sharpness/contrast metric


class FaceCropper:
    """Robust face cropper leveraging person detection boxes and facial keypoints."""

    def __init__(self, min_face_dim_px: int = 24, margin_ratio: float = 0.25):
        self.min_face_dim_px = min_face_dim_px
        self.margin_ratio = margin_ratio

    def crop(
        self,
        frame: np.ndarray,
        person_bbox: Tuple[float, float, float, float],
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray
    ) -> FaceCropResult:
        """
        Extracts face crop from student using upper facial keypoints (0..4)
        or top portion of person box as fallback.
        """
        frame_h, frame_w = frame.shape[:2]
        px1, py1, px2, py2 = person_bbox

        # Try keypoints first (nose, left_eye, right_eye, left_ear, right_ear)
        head_conf = keypoints_conf[:5]
        valid_head_kpts = [keypoints_xy[i] for i in range(5) if head_conf[i] > 0.35]

        if len(valid_head_kpts) >= 2:
            xs = [p[0] for p in valid_head_kpts]
            ys = [p[1] for p in valid_head_kpts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bw = max_x - min_x
            bh = max_y - min_y
            # Expand to include chin and forehead
            margin_x = max(8.0, bw * self.margin_ratio)
            margin_y = max(10.0, bh * (self.margin_ratio * 1.5))
            fx1 = max(0, int(min_x - margin_x))
            fy1 = max(0, int(min_y - margin_y))
            fx2 = min(frame_w, int(max_x + margin_x))
            fy2 = min(frame_h, int(max_y + margin_y))
        else:
            # Fallback: top 25% of person bounding box
            pw = px2 - px1
            ph = py2 - py1
            fx1 = max(0, int(px1 + 0.15 * pw))
            fy1 = max(0, int(py1))
            fx2 = min(frame_w, int(px2 - 0.15 * pw))
            fy2 = min(frame_h, int(py1 + 0.30 * ph))

        fw = fx2 - fx1
        fh = fy2 - fy1

        if fw < self.min_face_dim_px or fh < self.min_face_dim_px:
            return FaceCropResult(
                face_crop=None,
                bbox_px=(fx1, fy1, fx2, fy2),
                face_width=fw,
                face_height=fh,
                is_valid_resolution=False,
                quality_score=0.0
            )

        crop = frame[fy1:fy2, fx1:fx2].copy()

        # Compute sharpness quality (Laplacian variance)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        quality_score = float(np.clip(lap_var / 200.0, 0.0, 1.0))

        return FaceCropResult(
            face_crop=crop,
            bbox_px=(fx1, fy1, fx2, fy2),
            face_width=fw,
            face_height=fh,
            is_valid_resolution=True,
            quality_score=round(quality_score, 3)
        )
