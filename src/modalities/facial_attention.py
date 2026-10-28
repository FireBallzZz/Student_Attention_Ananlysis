"""Stage 3 — Facial and Postural Attention Modalities Module.

Extracts head pose, gaze proxies, torso posture, and movement energy from tracked students.
CRITICAL RESEARCH CONSTRAINT:
Explicitly gates distance and resolution. If a student is distant, occluded, or low-resolution,
facial landmarks, gaze, and expressions are marked UNRELIABLE and invalid instead of fabricated.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from src.tracking.tracker import TrackedStudent


@dataclass
class StudentModalities:
    """Multimodal attention features and reliability flags for one student frame."""
    track_id: int
    face_detected: bool
    face_confidence: float
    face_bbox: Optional[Tuple[float, float, float, float]]
    is_distant_student: bool

    # Head Pose (in degrees)
    head_pitch_deg: float      # negative: looking down (desk/notes), positive: looking up (board)
    head_yaw_deg: float        # negative: turned left, positive: turned right
    head_roll_deg: float       # tilt sideways
    head_pose_confidence: float
    head_pose_valid: bool

    # Gaze proxy
    gaze_horizontal: float     # visual focus vector proxy
    gaze_vertical: float
    gaze_valid: bool

    # Posture and Body
    torso_angle_deg: float     # ~90 deg is upright, <70 deg is slumped/leaning
    shoulder_tilt_deg: float   # degrees difference from level shoulders
    posture_valid: bool

    # Motion Dynamics
    movement_energy: float     # velocity of keypoint movement

    # Explicit Missingness Schema
    modality_missing_flags: Dict[str, bool] = field(default_factory=dict)


class ModalityExtractor:
    """Extracts multimodal behavioral indicators with honest uncertainty gating."""

    def __init__(
        self,
        min_face_size_px: float = 24.0,
        min_keypoint_conf: float = 0.35,
        min_student_area_ratio: float = 0.008
    ):
        self.min_face_size_px = min_face_size_px
        self.min_keypoint_conf = min_keypoint_conf
        self.min_student_area_ratio = min_student_area_ratio

    def extract(self, student: TrackedStudent, frame_shape: Tuple[int, int]) -> StudentModalities:
        """
        Extract observable modalities from tracked student.
        Explicitly identifies missing / unreliable modalities.
        """
        frame_h, frame_w = frame_shape[:2]
        kpts_xy = student.keypoints_xy
        kpts_conf = student.keypoints_conf

        # 1. Distance & Resolution Assessment
        is_distant = (student.area_ratio < self.min_student_area_ratio) or student.is_occluded

        face_detected = False
        face_conf = 0.0
        face_w, face_h = 0.0, 0.0

        if student.face_bbox is not None:
            fx1, fy1, fx2, fy2 = student.face_bbox
            face_w = fx2 - fx1
            face_h = fy2 - fy1
            head_kpts_conf = [kpts_conf[i] for i in range(5)]
            face_conf = float(np.mean(head_kpts_conf))
            face_detected = (face_conf >= self.min_keypoint_conf) and (min(face_w, face_h) >= self.min_face_size_px)

        missing_flags = {
            "face_landmarks": not face_detected or is_distant,
            "head_pose": False,
            "gaze": not face_detected or is_distant,
            "torso_posture": False,
            "facial_expression": not face_detected or is_distant
        }

        # 2. Head Pose Estimation (from keypoints: nose(0), l_eye(1), r_eye(2), l_ear(3), r_ear(4))
        head_pitch = 0.0
        head_yaw = 0.0
        head_roll = 0.0
        head_pose_conf = 0.0
        head_pose_valid = False

        if face_detected and not is_distant:
            nose = kpts_xy[0]
            l_eye = kpts_xy[1]
            r_eye = kpts_xy[2]
            l_ear = kpts_xy[3]
            r_ear = kpts_xy[4]

            eye_center = (l_eye + r_eye) / 2.0
            eye_dist = np.linalg.norm(r_eye - l_eye) + 1e-6

            # Roll: slope of eye line
            eye_dy = r_eye[1] - l_eye[1]
            eye_dx = r_eye[0] - l_eye[0]
            head_roll = float(np.degrees(np.arctan2(eye_dy, eye_dx)))

            # Yaw: asymmetry between nose-to-left-ear vs nose-to-right-ear
            if kpts_conf[3] > self.min_keypoint_conf and kpts_conf[4] > self.min_keypoint_conf:
                d_l = np.linalg.norm(nose - l_ear)
                d_r = np.linalg.norm(nose - r_ear)
                yaw_ratio = (d_r - d_l) / (d_r + d_l + 1e-6)
                head_yaw = float(np.clip(yaw_ratio * 90.0, -90.0, 90.0))
            else:
                # Fallback to nose-eye horizontal offset
                yaw_offset = (nose[0] - eye_center[0]) / eye_dist
                head_yaw = float(np.clip(yaw_offset * 60.0, -60.0, 60.0))

            # Pitch: vertical displacement of nose relative to eye level
            # When looking down at desk, nose drops further down relative to eye distance
            pitch_ratio = (nose[1] - eye_center[1]) / eye_dist
            # baseline neutral ratio is ~0.6; higher means bowing head/looking down
            head_pitch = float(np.clip((0.6 - pitch_ratio) * 60.0, -60.0, 60.0))

            head_pose_conf = float(np.mean([kpts_conf[i] for i in range(5)]))
            head_pose_valid = head_pose_conf >= 0.4
        else:
            missing_flags["head_pose"] = True

        # 3. Gaze Proxy (Direction of visual attention)
        gaze_h = 0.0
        gaze_v = 0.0
        gaze_valid = False

        if face_detected and head_pose_valid and not is_distant:
            # Under standard desk viewing, gaze roughly aligns with head yaw and pitch
            gaze_h = float(head_yaw / 90.0)    # range: [-1, 1]
            gaze_v = float(head_pitch / 90.0)  # range: [-1, 1]
            gaze_valid = True
        else:
            missing_flags["gaze"] = True

        # 4. Posture and Body Mechanics
        # Shoulders: l_shoulder(5), r_shoulder(6)
        # Hips: l_hip(11), r_hip(12)
        torso_angle = 90.0  # default upright
        shoulder_tilt = 0.0
        posture_valid = False

        if kpts_conf[5] > self.min_keypoint_conf and kpts_conf[6] > self.min_keypoint_conf:
            l_sh = kpts_xy[5]
            r_sh = kpts_xy[6]
            sh_center = (l_sh + r_sh) / 2.0
            sh_dy = r_sh[1] - l_sh[1]
            sh_dx = r_sh[0] - l_sh[0]
            shoulder_tilt = float(np.degrees(np.arctan2(sh_dy, sh_dx)))

            if kpts_conf[11] > self.min_keypoint_conf and kpts_conf[12] > self.min_keypoint_conf:
                hip_center = (kpts_xy[11] + kpts_xy[12]) / 2.0
                spine_dy = hip_center[1] - sh_center[1]
                spine_dx = sh_center[0] - hip_center[0]
                torso_angle = float(np.degrees(np.arctan2(spine_dy, np.abs(spine_dx) + 1e-6)))
                posture_valid = True
            else:
                posture_valid = True  # shoulder tilt available even if hips occluded by desk
        else:
            missing_flags["torso_posture"] = True

        # 5. Motion Energy
        movement_energy = float(np.linalg.norm(student.velocity))

        return StudentModalities(
            track_id=student.track_id,
            face_detected=face_detected,
            face_confidence=round(face_conf, 3),
            face_bbox=student.face_bbox if face_detected else None,
            is_distant_student=is_distant,
            head_pitch_deg=round(head_pitch, 2),
            head_yaw_deg=round(head_yaw, 2),
            head_roll_deg=round(head_roll, 2),
            head_pose_confidence=round(head_pose_conf, 3),
            head_pose_valid=head_pose_valid,
            gaze_horizontal=round(gaze_h, 3),
            gaze_vertical=round(gaze_v, 3),
            gaze_valid=gaze_valid,
            torso_angle_deg=round(torso_angle, 2),
            shoulder_tilt_deg=round(shoulder_tilt, 2),
            posture_valid=posture_valid,
            movement_energy=round(movement_energy, 2),
            modality_missing_flags=missing_flags
        )
