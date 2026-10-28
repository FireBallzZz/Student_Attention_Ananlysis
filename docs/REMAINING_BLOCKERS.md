# Remaining Blockers & Research Limitations

**Thesis Project:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis, Pedagogical Recommendation, and Learning Improvement Assessment in Smart Classrooms  
**Status Date:** 2026-10-28  

---

## 1. System & Dependency Environment Blockers

### 1.1 FFmpeg Binary in Windows PATH
- **Issue:** The `ffmpeg` CLI executable is not installed in the system PATH.
- **Impact:** Direct H.264 video re-muxing via `ffmpeg` subprocesses is unavailable.
- **Applied Mitigation:** The pipeline utilizes OpenCV's native `mp4v` codec with automatic fallback. Videos render and play reliably in modern HTML5 browsers.
- **Recommended Future Action:** For high compression and native H.264 streaming across legacy mobile clients, install FFmpeg (`winget install Gyan.FFmpeg` or manual binary download) and add it to system PATH.

---

## 2. Scientific & Dataset Limitations

### 2.1 Clip 11 Ground-Truth Annotations
- **Issue:** `data/clip_011.mp4` serves as the independent out-of-distribution evaluation clip, but currently lacks frame-by-frame CVAT individual attention labels.
- **Scientific Impact:** We can execute inference, compute census tracking, extract 16D features, and assess before-and-after behavioral shifts on Clip 11, but formal quantitative confusion matrices on Clip 11 require ground-truth labels.
- **Recommended Action:** Annotate Clip 11 with the established 3-class standard (High, Moderate, Low) across visible students following ethical protocols.

### 2.2 Rear-Row Facial Occlusion & Low Pixel Resolution
- **Issue:** Students seated in rows 4+ in elevated wide-angle cameras occupy $< 1\%$ of frame area, resulting in facial bounding boxes $< 24\times24$ pixels.
- **Scientific Mitigation:** The pipeline explicitly gates distant students with `is_distant=1.0` and assigns `Uncertain` rather than fabricating arbitrary gaze angles.
- **Future Improvement:** Incorporating multi-camera views (e.g., front-row + rear-angle camera fusion) will improve coverage for distant students.

---

## 3. Scope Exclusions (As Defined by Master Prompt)

### 3.1 Large Language Model (LLM) Integration
- **Status:** **Completely excluded in this phase.**
- **Details:** No paid APIs, external LLM endpoints, or mock LLM recommendation generators have been introduced.
- **Prepared Foundation:** The structured JSON telemetry, drop event triggers, and before-and-after assessment results are ready for prompt-engineering and local LLM ingestion in the subsequent research phase.
