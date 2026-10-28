# Setup and Execution Guide

**Project:** Smart Classroom Attention Analytics (Non-LLM Core System)  
**Target Environment:** Windows 10/11 or Linux, Python 3.10+  

---

## 1. Prerequisites and Installation

### Hardware Acceleration (Optional but Recommended)
- NVIDIA GPU with CUDA support (e.g., RTX 3060). The pipeline automatically falls back to CPU if CUDA is unavailable.

### Clone and Setup
```powershell
# Navigate to project repository
cd "C:\Users\User\Downloads\Model v2\classroom_attention_research"

# Ensure dependencies are installed
pip install -r requirements.txt
```

---

## 2. Running the Interactive Web Application

Launch the FastAPI backend server:
```powershell
python web_app/server.py
```

The server initializes on port 8000:
- **Web UI:** Open your browser and navigate to: `http://127.0.0.1:8000`
- **Interactive Features:**
  - Select pre-loaded pilot clips (`View 1`, `View 3`, `View 11`), ground-truth benchmark (`Clip 4`), or independent test (`Clip 11`).
  - Upload custom classroom MP4/AVI recordings with automatic file integrity validation.
  - Configure processing duration (3 to 15 seconds) and run attention analysis.
  - Stream annotated diagnostic video with HTTP 206 Partial Content support.
  - Review executive telemetry (census count, active vs coasting tracks, high/moderate/low/uncertain distributions).
  - Inspect Non-LLM Learning Improvement Assessment (Welch's t-test, Cohen's d, observable behavioral shift table).
  - Interactively recompute interval statistics with custom temporal split timestamps.
  - Export output video and comprehensive telemetry JSON.

---

## 3. Running Video Inference via Python Command Line

To process a video directly from the command line:

```powershell
python -c "
from pathlib import Path
from src.pipeline.video_pipeline import ClassroomAttentionPipeline

pipeline = ClassroomAttentionPipeline(sample_fps=10.0, drop_threshold=20.0)
results = pipeline.process_video(
    input_video_path=Path('data/clip_011.mp4'),
    output_video_path=Path('web_app/static/outputs/annotated_clip_011.mp4'),
    max_frames=100
)
print('Inference Complete!')
print('Census Students:', results['census_student_count'])
print('Average Attention:', results['class_average_attention'])
print('Breakdown:', results['observation_breakdown'])
"
```

---

## 4. Running the Test Suite

Execute the full automated test suite (31 unit and integration tests):

```powershell
python -m pytest tests/
```

To run individual test modules:
```powershell
# Video ingestion and validation tests
python -m pytest tests/test_video_ingestion.py

# Tracking contract and diagnostics tests
python -m pytest tests/test_tracking_contract.py

# Multimodal feature schema tests
python -m pytest tests/test_feature_schema.py

# Real model inference and class mapping tests
python -m pytest tests/test_model_inference.py

# Non-LLM Before-and-after assessment tests
python -m pytest tests/test_before_after.py

# FastAPI backend endpoint tests
python -m pytest tests/test_server_endpoints.py

# End-to-end pipeline smoke test
python -m pytest tests/test_end_to_end_smoke.py
```
