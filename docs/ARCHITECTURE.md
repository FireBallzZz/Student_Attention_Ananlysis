# System Architecture & Pipeline Specification

**Title:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis, Pedagogical Recommendation, and Learning Improvement Assessment in Smart Classrooms  
**Subsystem:** Core Computer Vision & Machine Learning Pipeline (Non-LLM Foundation)  
**Version:** 1.0.0 Production Foundation  

---

## 1. High-Level Architectural Overview

The framework provides an end-to-end, reproducible, non-LLM computer vision and machine learning pipeline for real-time classroom video analysis. It ingests high-definition classroom recordings, performs multi-student pose estimation and persistent identity tracking, extracts a 16-dimensional physical multimodal feature schema, evaluates attention via a supervised ensemble model, renders an annotated telemetry video, and computes statistical learning improvement metrics across temporal intervals.

```
+----------------------------------------------------------------------------------------------------+
|                                    Classroom Video Ingestion                                       |
|                (cv2.VideoCapture, metadata extraction, corruption & 0-byte validation)             |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                               Student Detection & Pose Keypoints                                   |
|                      (YOLO11n-Pose: 17 COCO Keypoints, Face BBox, Area Ratio)                      |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                                 Classroom Census & Memory Tracking                                 |
|          (ClassroomCensusTracker: IoU Association, EMA Smoothing, Edge Exit vs Coasting)           |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                                Multimodal Modality Extraction                                      |
|      (Head Pitch/Yaw/Roll, Gaze Proxies, Torso Inclination, Motor Energy, Gating Flags)            |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                               Standardized 16D Feature Vector                                      |
|              (Continuous Observables + Explicit Binary Missingness/Distance Flags)                 |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                            Calibrated Attention Classification                                     |
|             (RandomForestClassifier: P(High), P(Moderate), P(Low) -> Calibrated 0-100)             |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                         +-------------------------+-------------------------+
                         |                                                   |
                         v                                                   v
+--------------------------------------------------+ +-----------------------------------------------+
|         Diagnostic Video HUD & Export            | |    Non-LLM Learning Improvement Assessment    |
|   (BBoxes, Dynamic HUD Bar, MP4V Video Writer)   | |  (Welch's t-test, Cohen's d, Behavioral Shift)|
+--------------------------------------------------+ +-----------------------------------------------+
                         |                                                   |
                         +-------------------------+-------------------------+
                                                   |
                                                   v
+----------------------------------------------------------------------------------------------------+
|                               FastAPI Server & Interactive Dashboard                               |
|        (REST API, HTTP 206 Partial Content Video Streaming, Real-Time Recomputation)              |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Component Breakdown

### Stage 1: Ingestion & Integrity Validation (`src/ingestion/video_reader.py`)
- **Purpose:** Decodes video streams frame-by-frame using a streaming generator pattern to eliminate OOM risks on long classroom lectures.
- **Validation:** Inspects dimensions, FPS (valid range: 1.0–240.0), frame count, fourcc codec, file existence, non-zero file size, and verifies first frame decodability.
- **Contract:** Emits `TimestampedFrame(frame_idx, timestamp_sec, frame, clip_id, session_id)`.

### Stage 2: Pose Detection & Student Census Tracking (`src/tracking/tracker.py`)
- **Detector:** Uses Ultralytics YOLO11-Pose to locate students and identify 17 COCO keypoints (eyes, nose, shoulders, hips, limbs).
- **Tracker:** `ClassroomCensusTracker` solves 5 classroom-specific challenges:
  1. *Student Census:* Establishes total seated room capacity during initial frames.
  2. *Temporal Memory Reference (Coasting):* Prevents lost identities when students tilt heads down to read or write by referencing prior bounding boxes.
  3. *De-Duplication Suppression:* Suppresses redundant detections with IoU gating (`iou_dedup_thresh=0.45`).
  4. *Out-of-Frame Exit Detection:* Distinguishes genuine boundary exit (`near_edge` threshold) from temporary occlusion inside the room.
  5. *Box Dynamics Telemetry:* Emits runtime status updates (`INCREASE`, `VANISHED`, `STUDENT_EXITED_FRAME`, `STABLE`).
- **Contract:** Outputs `List[TrackedStudent]`.

### Stage 3: Multimodal Feature Extraction (`src/modalities/facial_attention.py`)
- **Physical Observables:**
  - *Head Pose:* Pitch (up/down), Yaw (left/right rotation), Roll (lateral tilt), and estimation confidence.
  - *Eye Gaze Proxy:* Horizontal/vertical deviations relative to facial midpoint.
  - *Torso Posture:* Angle between mid-shoulder and mid-hip relative to gravity; shoulder tilt.
  - *Motor Dynamics:* Frame-to-frame bounding box movement velocity.
- **Epistemic Distance Gating:** Students with bounding box area ratio `< 0.01` (distant rows with <24px facial width) are explicitly flagged with `is_distant=1.0` and `missing_face=1.0` rather than fabricating arbitrary facial angles.

### Stage 4: Feature Vectorization & Ordering (`src/features/feature_schema.py`)
- Strictly enforces a 16-dimensional column schema matching `checkpoints/final_attention_model.joblib`:
  `[head_pitch_deg, head_yaw_deg, head_roll_deg, head_pose_confidence, gaze_horizontal, gaze_vertical, torso_angle_deg, shoulder_tilt_deg, movement_energy, bbox_area_ratio, detection_confidence, missing_face, missing_head_pose, missing_gaze, missing_torso, is_distant]`.

### Stage 5: Attention Classification (`src/models/attention_scorer.py`)
- **Production Model:** Loads `RandomForestClassifier` from `checkpoints/final_attention_model.joblib`.
- **Class Mapping:** `['High', 'Low', 'Moderate']`.
- **Continuous Score Calibration:**
  $$\text{Score} = \text{clip}\left(100.0 \times \left(P(\text{High}) + 0.5 \times P(\text{Moderate})\right), 5.0, 98.0\right)$$
- **Uncertainty Assignment:** If missing flags $\ge 2$ or `is_distant == 1.0`, the system outputs `category = "Uncertain"`, preserving scientific truth over forced predictions.

### Stage 6: Learning Improvement Assessment (`src/evaluation/before_after_analysis.py`)
- Evaluates observable engagement differences across pedagogical intervals (e.g. baseline vs post-intervention).
- Computes:
  - High, Moderate, Low, and Uncertain observation counts and percentages.
  - Observation coverage and missingness rates.
  - Welch's two-sample $t$-test (unequal variances assumed) and $p$-value.
  - Cohen's $d$ effect size:
    $$d = \frac{\bar{X}_2 - \bar{X}_1}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$
  - Observable physical behavioral shifts (% points for Board Focus, Desk/Reading, Slumped Posture, Turned Away, Restless Movement).
- **Epistemic Boundary:** Formally documents that attention scores reflect observable physical behavioral indicators and do not directly measure internal cognitive learning gains.

### Stage 7: Full-Stack Interface (`web_app/server.py` & `web_app/static/`)
- FastAPI asynchronous backend supporting video upload, sample video selection, streaming playback with HTTP 206 Partial Content, telemetry exports (JSON), and interactive interval comparison.
- Modern glassmorphism UI styled with vanilla CSS and responsive JavaScript.

---

## 3. Exclusion of LLM & Future Modular Integration Point

In accordance with thesis milestones:
- **Current State:** LLM endpoints, providers, API keys, and external calls are **100% excluded**.
- **Future Integration Point:** The `src/recommendations/recommendation_engine.py` module exposes a clean abstract interface `generate_recommendation(drop_event, context)`. When LLM capabilities are introduced, this module can query a local LLM or provider using the structured JSON telemetry generated by `ClassroomAttentionPipeline` without altering any computer vision, tracking, or feature extraction code.
