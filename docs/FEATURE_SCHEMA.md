# Multimodal Feature Schema & Vectorization Specification

**Module:** `src/features/feature_schema.py` & `src/modalities/facial_attention.py`  
**Checkpoint Invariance:** Must match `checkpoints/final_attention_model.joblib` (`n_features_in_ == 16`)  

---

## 1. Canonical 16-Dimensional Feature Schema

The classifier accepts a 16-dimensional float32 vector per student observation. The order of columns is strictly enforced:

| Index | Feature Name | Data Type | Physical Unit | Valid Range | Description & Extraction Method | Missing-Value Behavior |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | `head_pitch_deg` | `float32` | Degrees | $[-90.0, 90.0]$ | Vertical head tilt (negative = head down/writing, positive = looking up at board). Derived from nose-to-ear vertical displacement. | Defaults to `0.0` when facial keypoints are occluded or distant. |
| **1** | `head_yaw_deg` | `float32` | Degrees | $[-90.0, 90.0]$ | Horizontal head rotation (negative = left, positive = right). Derived from relative ear-to-nose horizontal displacement. | Defaults to `0.0` when face is occluded or distant. |
| **2** | `head_roll_deg` | `float32` | Degrees | $[-45.0, 45.0]$ | Lateral head tilt (ear-to-shoulder angle). | Defaults to `0.0` when face is occluded or distant. |
| **3** | `head_pose_confidence`| `float32` | Normalized | $[0.0, 1.0]$ | Mean detection confidence of facial keypoints (nose, eyes, ears). | Defaults to `0.0` when face is unavailable. |
| **4** | `gaze_horizontal` | `float32` | Normalized | $[-1.0, 1.0]$ | Proxy eye gaze direction relative to facial center (negative = left, positive = right). | Defaults to `0.0` when eyes are occluded or distant. |
| **5** | `gaze_vertical` | `float32` | Normalized | $[-1.0, 1.0]$ | Proxy eye gaze direction relative to facial center (negative = down, positive = up). | Defaults to `0.0` when eyes are occluded or distant. |
| **6** | `torso_angle_deg` | `float32` | Degrees | $[0.0, 180.0]$ | Inclination angle between mid-shoulder and mid-hip relative to gravity (80°–95° = upright, <60° = slumped). | Defaults to `90.0` (neutral upright) when hips are occluded by desk. |
| **7** | `shoulder_tilt_deg`| `float32` | Degrees | $[-45.0, 45.0]$ | Angle of shoulder line relative to horizontal plane. | Defaults to `0.0` when shoulders are occluded. |
| **8** | `movement_energy` | `float32` | Pixels/sec | $[0.0, \infty)$ | Magnitude of bounding box centroid displacement between consecutive sampled frames. | Defaults to `0.0` on first frame. |
| **9** | `bbox_area_ratio` | `float32` | Ratio | $[0.0, 1.0]$ | Bounding box area divided by full frame area $(w \times h)$. Indicates student distance and scale. | Exact ratio from tracker bounding box. |
| **10**| `detection_confidence`| `float32`| Normalized | $[0.0, 1.0]$ | YOLO11 person detection confidence score. | Direct detector output. |
| **11**| `missing_face` | `float32` | Binary flag | $\{0.0, 1.0\}$ | Explicit indicator: `1.0` if facial keypoints confidence $< 0.4$ or student is distant, `0.0` otherwise. | Binary explicit flag. |
| **12**| `missing_head_pose`| `float32` | Binary flag | $\{0.0, 1.0\}$ | Explicit indicator: `1.0` if head orientation could not be reliably estimated, `0.0` otherwise. | Binary explicit flag. |
| **13**| `missing_gaze` | `float32` | Binary flag | $\{0.0, 1.0\}$ | Explicit indicator: `1.0` if eye keypoints confidence $< 0.4$ or student is distant, `0.0` otherwise. | Binary explicit flag. |
| **14**| `missing_torso` | `float32` | Binary flag | $\{0.0, 1.0\}$ | Explicit indicator: `1.0` if shoulder or hip keypoints are occluded by desks, `0.0` otherwise. | Binary explicit flag. |
| **15**| `is_distant` | `float32` | Binary flag | $\{0.0, 1.0\}$ | Explicit indicator: `1.0` if `bbox_area_ratio < 0.01` (facial region $< 24\text{px}$), `0.0` otherwise. | Binary explicit flag. |

---

## 2. Epistemic Distance Gating Rule

When `bbox_area_ratio < 0.01` (representing distant rows in large lecture halls):
- `is_distant` is set to `1.0`.
- `missing_face`, `missing_head_pose`, and `missing_gaze` are set to `1.0`.
- Continuous facial features are set to neutral zeros.
- The classifier assigns category `Uncertain` rather than fabricating arbitrary gaze or facial angles.

---

## 3. Vectorization Code

To produce the numpy feature vector:
```python
from src.features.feature_schema import create_feature_vector

fv = create_feature_vector(
    session_id="session_01",
    clip_id="clip_004",
    student=tracked_student,
    modalities=extracted_modalities,
    frame_idx=frame_idx,
    timestamp_sec=timestamp_sec
)
feature_array = fv.to_feature_array() # shape: (16,), dtype: float32
```
