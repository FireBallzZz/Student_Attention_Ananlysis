# Test Execution & Verification Report

**Date of Execution:** 2026-10-28  
**Environment:** Windows 10/11, Python 3.14.4, PyTorch 2.14.1+cu126 (NVIDIA RTX 3060 Laptop GPU)  
**Test Framework:** Pytest 8.3.4 with AnyIO & Pytest-AsyncIO  
**Status:** **ALL 31 TESTS PASSED (100% Pass Rate)**  

---

## 1. Test Suite Summary Table

| Test Module | Coverage Area | Number of Tests | Status | Execution Mode |
| :--- | :--- | :---: | :---: | :--- |
| `tests/test_video_ingestion.py` | Ingestion, validation, 0-byte detection, stream generator | 5 | **PASSED** | Real OpenCV Video Streams |
| `tests/test_tracking_contract.py` | Keypoint contract, persistent track IDs, coasting, exit detection | 4 | **PASSED** | Real `ClassroomCensusTracker` logic |
| `tests/test_feature_schema.py` | 16D column schema, feature ordering, distance gating flags | 3 | **PASSED** | Real `create_feature_vector` logic |
| `tests/test_model_inference.py` | Checkpoint verification, class order, probability calibration | 5 | **PASSED** | **REAL PRODUCTION MODEL CHECKPOINT** |
| `tests/test_before_after.py` | Welch's t-test, Cohen's d, High/Mod/Low proportions, shifts | 3 | **PASSED** | Real Scipy / Statistical calculations |
| `tests/test_server_endpoints.py` | Index, samples, upload validation, interval comparison endpoint | 6 | **PASSED** | Real FastAPI TestClient |
| `tests/test_end_to_end_smoke.py` | Video -> Detection -> Tracking -> Scoring -> Render -> Report | 1 | **PASSED** | **REAL FULL PIPELINE EXECUTION** |
| `tests/test_pipeline.py` | Distance gating, drop detection, recommendation contract | 4 | **PASSED** | Unit & Component Integration |
| **Total** | **Full System Verification** | **31** | **PASSED** | **100% Operational** |

---

## 2. Verification Mode Breakdown

### 2.1 Components Verified Using the REAL Model Checkpoint
- **`checkpoints/final_attention_model.joblib`:**
  - Verified model architecture is `RandomForestClassifier`.
  - Verified `n_features_in_ == 16`.
  - Verified class order is `['High', 'Low', 'Moderate']`.
  - Verified real inference produces valid probabilities summing to 1.0.
  - Verified continuous calibrated score within $[5.0, 98.0]$.
  - Verified attentive vs inattentive discriminating capability.

### 2.2 Components Verified Using Real System Primitives
- **OpenCV Video Codecs:** Verified native frame writing using `mp4v` codec with valid resolution and playback headers.
- **YOLO11-Pose Model:** Verified keypoint extraction and person bounding box generation in end-to-end smoke testing.
- **FastAPI Endpoints:** Verified synchronous and asynchronous route handlers, HTTP 206 partial streaming headers, and error code propagation.

### 2.3 Components Tested with Controlled Fixtures / Mocks
- **Corrupt File Fixtures:** Synthetic 0-byte and corrupted byte sequences created in temporary directories to verify exception safety.
- **Sparse Observation Edge Cases:** Controlled synthetic timelines used to verify Welch's $t$-test fallback when $N < 2$.

---

## 3. Actual Execution Log Output

```text
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-8.3.4, pluggy-1.6.0
rootdir: C:\Users\User\Downloads\Model v2\classroom_attention_research
plugins: anyio-4.15.1, asyncio-0.25.2
asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None
collected 31 items

tests\test_before_after.py ...                                           [  9%]
tests\test_end_to_end_smoke.py .                                         [ 12%]
tests\test_feature_schema.py ...                                         [ 22%]
tests\test_model_inference.py .....                                      [ 38%]
tests\test_pipeline.py ....                                              [ 51%]
tests\test_server_endpoints.py ......                                    [ 70%]
tests\test_tracking_contract.py ....                                     [ 83%]
tests\test_video_ingestion.py .....                                      [100%]

======================= 31 passed, 33 warnings in 5.32s =======================
```
