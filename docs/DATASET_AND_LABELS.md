# Dataset, Annotations, and Ground-Truth Labeling Specification

**Thesis Title:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis, Pedagogical Recommendation, and Learning Improvement Assessment in Smart Classrooms  
**Document:** Dataset Integrity, Annotation Standards, and Scientific Grounding  

---

## 1. Classroom Video Corpus

The project repository utilizes five verified classroom recordings representing distinct camera angles, lighting conditions, and student densities:

| Video Identifier | File Location | Resolution | Duration | FPS | Observed Density | Annotation Status | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clip 4** | `data/clip 4.mp4` | 1920x1080 | 10.0s (300 f) | 30.0 | 21 seated students | Full CVAT individual annotations | Model Training & Internal Benchmark |
| **Clip 11** | `data/clip_011.mp4` | 1920x1080 | 10.0s (300 f) | 30.0 | 18–20 students | Unseen independent benchmark | **Independent Out-of-Sample Evaluation** |
| **View 1** | `data/pilot/high_engagement/high/view1.mp4` | 1920x1080 | 11.5s (345 f) | 30.0 | 6–8 front students | Pilot baseline | Perspective Benchmark (High Engagement) |
| **View 3** | `data/pilot/low_engagement/low/view3.mp4` | 1920x1080 | 13.5s (405 f) | 30.0 | 8–12 side students | Pilot baseline | Oblique Perspective (Low Engagement) |
| **View 11** | `data/pilot/high_engagement/high/view11.mp4` | 1920x1080 | 11.5s (345 f) | 30.0 | 10–15 students | Pilot baseline | Rear High-Angle (Distance Gating) |

---

## 2. Ground-Truth Annotations for Clip 4

### 2.1 File Location & Structure
- **Annotation Source:** `data/annotate for clip 4/annotations.xml`
- **Format:** CVAT XML for video 1.1 format.
- **Track Count:** 21 tracks (labeled `student_1` through `student_21`).
- **Total Labeled Boxes:** 6,300 bounding box instances (21 tracks $\times$ 300 frames).

### 2.2 Individual Attention Attribute Schema
Each bounding box in CVAT contains an explicit `<attribute name="attention">` tag:
- `<attribute name="attention">High</attribute>`
- `<attribute name="attention">Moderate</attribute>`
- `<attribute name="attention">Low</attribute>`

### 2.3 Individual vs. Group-Level Ground Truth
**Scientific Rule:** The pipeline assigns attention on an **individual student basis** using specific physical markers for that student. Group-level aggregate engagement labels are NEVER applied as individual ground truth.

---

## 3. Data Leakage & Evaluation Methodology

### 3.1 Empirical Audit of Checkpoint Origin
An independent audit of `checkpoints/final_attention_model.joblib` confirmed:
1. The Random Forest classifier was fitted on 5,732 feature vectors extracted directly from `data/clip 4.mp4`.
2. The reported 93.3% accuracy on Clip 4 represents **train-set re-substitution**, not cross-dataset generalization.

### 3.2 Evaluation Protocol for Generalization
To maintain scientific validity:
- **Clip 4** is documented as the reference training and internal consistency benchmark.
- **Clip 11** (`data/clip_011.mp4`) is designated as the **unseen evaluation set**. It has not been used to train, tune hyperparameters, or calibrate decision thresholds.
- Metrics reported on Clip 4 must not be cited as proof of independent generalization.

---

## 4. Safety & Research Integrity Constraints

1. **Exclusion of `PRE DEFENCE/YOLO Test.mp4`:**
   - Under no circumstances is `PRE DEFENCE/YOLO Test.mp4` to be used for model training, tuning, or benchmark reporting.
2. **Preservation of Artifacts:**
   - Ground-truth files (`annotations.xml`), raw videos, and existing reports must not be modified or overwritten.
3. **No Synthetic Training Data:**
   - Synthetic fixtures are used strictly for software unit tests and never substituted as genuine empirical research data.
4. **Epistemic Distinction:**
   - Attention scores measure observable physical behaviors (head orientation, gaze proxy, torso posture, and motor dynamics). They do not constitute direct cognitive measurement of internal learning comprehension.
