# Minimal Individual-Attention Pilot Plan & Defensible Alternative

**Date**: October 27, 2026  
**Auditor**: Senior Computer Vision Research Engineer & ML Scientist  
**Repository**: `c:\Users\User\Downloads\Model v2\classroom_attention_research`  
**Guiding Principle**: Legitimate individual labels, robust generalization, and honest evaluation over inflated accuracy, model complexity, or dashboard appearance.

---

## 1. Candidate Ranking Against Primary Research Objective

Our primary research objective requires:
1. Raw multi-student continuous video in a physical classroom.
2. Student-specific annotations linked to visible individuals over time.
3. Temporal continuity for multi-object tracking (MOT) feasibility.
4. Open, legitimate reuse rights.
5. Environmental and behavioral variation across seating and camera angles.

### Objective Ranking Matrix

| Candidate | Multi-Student Raw Video | Individual Student GT | Temporal Tracking Continuity | Legal Reuse Rights | Rank | Strategic Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **OUC-CGE (Pilot)** | **10/10** (Continuous 30 FPS, 12 cameras) | **0/10** (Group labels only) | **10/10** (DeepSORT 73.8 FPS, persistent IDs) | **10/10** (CC BY 4.0 Open Access) | **#1 (Pipeline Engine)** | Full technical CV tracking, detection, distance gating, and behavioral scoring engine. |
| **SCB-Dataset** | **5/10** (Static classroom images, not video) | **9/10** (BBoxes for reading, writing, head-bowing) | **2/10** (Static frames, no track sequences) | **8/10** (Academic / OpenDataLab) | **#2 (Action Calibration)** | Benchmark individual physical action detection without claiming cognitive mental states. |
| **Mendeley `smzggbnkd2/3`** | **0/10** (CSV only, zero video/images) | **8/10** (Individual attention labels in tabular form) | **0/10** (Zero video frames to track) | **10/10** (CC BY 4.0 Open Access) | **#3 (Tabular Baseline)** | Tabular feature classification benchmark for webcam features only. |
| **DAiSEE** | **6/10** (10s clips, single user webcam) | **9/10** (Individual affective states) | **1/10** (Single user; zero classroom tracking) | **5/10** (Restricted EULA form) | **#4 (Affective Baseline)** | Facial expression & gaze feature calibration in e-learning contexts. |

---

## 2. The Defensible Alternative Architecture

Because **no off-the-shelf public dataset provides continuous multi-student classroom video AND trustworthy individual attention labels simultaneously**, the only scientifically defensible path that avoids committing an ecological fallacy consists of a **Two-Tiered Hybrid Validation Framework**:

```mermaid
flowchart TD
    subgraph Tier1["Tier 1: Continuous Vision Pipeline (Existing OUC-CGE Pilot)"]
        V[Raw Multi-Student Video] --> Yolo[YOLO11-Pose Detection]
        Yolo --> DeepSORT[DeepSORT Tracking]
        DeepSORT --> Mod[Modality Extraction & Gating]
        Mod --> Scoring[Unsupervised Observable Attention Index 0–100]
        Scoring --> Drop[Attention Drop Detector]
        Drop --> LLM[Pedagogical Intervention Engine]
    end

    subgraph Tier2["Tier 2: Legitimate Individual Attention Ground Truth"]
        Annotation["Minimal Expert-Annotated Classroom Sample
        (3 Diverse Clips, 30s each: Front, Middle, Rear Rows)"]
        Annotation --> Coders["Standardized Protocol:
        • Active Board/Teacher Gaze
        • Note-Taking / Reading
        • Head Slump / Sleep
        • Off-Task Talking / Disengagement
        (Krippendorff's Alpha Inter-Coder Reliability)"]
        Coders --> Benchmarking["Rigorous Individual Attention Benchmarking:
        • Detection Precision/Recall
        • MOTA / IDF1 Tracking Continuity
        • Individual Attention Balanced Acc & F1
        • Coverage & Explicit UNCERTAIN Outcomes"]
    end

    Tier1 -.-> Benchmarking
```

### Why This Alternative is Defensible & Superior:
1. **Never commits an ecological fallacy**: Group-level labels are never mapped onto individual students.
2. **Never invents raw video**: Does not misrepresent tabular datasets (like Mendeley) as classroom video.
3. **Scientifically Grounded**: Ground truth is established using a transparent, pre-registered educational coding protocol with inter-annotator agreement statistics (Krippendorff's $\alpha \ge 0.80$).
4. **Validates Unseen-Classroom Generalization**: Evaluates the pipeline on held-out camera views and diverse classroom seating zones (front row vs rear row distance gating).

---

## 3. Minimal Download & Inspection Protocol (Phase 3)

In accordance with the **Hard Stop directive**, we do **not** download the full 16.5 GB dataset. The minimal useful sample protocol is structured as follows:

### Step 1: Utilize Existing Verified Dynamic Pilot Footage (Zero Bytes Downloaded)
* **Status**: We already have **21 verified dynamic multi-camera clips** on disk passing 100% checksum audits.
* **Selection of Minimal Evaluation Set**:
  * **Clip A (`view1.mp4`, High Engagement)**: Front-center perspective, high illumination, 6–8 visible students.
  * **Clip B (`view3.mp4`, Low Engagement)**: Side-angle perspective, natural student slumping, 1080p resolution.
  * **Clip C (`view11.mp4`, High Engagement)**: Rear-perspective showing distant rear-row students ($<24\text{ px}$ face width).
* **Total Duration**: 3 clips $\times$ 10–15 seconds = ~35 seconds of footage (over 1,000 video frames).
* **Storage Footprint**: **0 MB additional download** (uses existing verified pilot).

### Step 2: Download Minimal Supplementary Action Benchmark (Only Upon Approval)
* If static individual action detection benchmarking is required:
  * Download **only** the test split of **SCB-Dataset** (~200 MB sample) directly from OpenDataLab.
  * Strictly avoid downloading the full 8 GB corpus.

---

## 4. Phase 4 — Pilot Feasibility Test Protocol

When the minimal individual-annotated sample is evaluated, the experiment must follow this strict scientific protocol:

### 1. Separation of Detection/Tracking Failures vs Classification Errors
* **Tracking Failures**:
  - False Negative Detections (missed students due to occlusion).
  - ID Switches (DeepSORT track ID transitions).
  - Unsupervised Track Continuity Ratio (UTDR).
* **Attention Classification Errors**:
  - Evaluated **only** on successfully tracked student instances where visual evidence is sufficient.
  - Instances where face $<24\text{ px}$ or occlusion $>50\%$ are explicitly assigned an **`UNCERTAIN` / `INSUFFICIENT_VISUAL_EVIDENCE`** outcome rather than forcing a fabricated guess.

### 2. Multi-Metric Evaluation Matrix
The evaluation must report:
* Per-Class Precision, Recall, and F1 (`Attentive`, `Inattentive`, `Uncertain`).
* Macro F1 and Balanced Accuracy.
* Confusion Matrix:
  $$\mathbf{C}_{\text{eval}} = \begin{pmatrix} C_{\text{inattentive},\text{inattentive}} & C_{\text{inattentive},\text{attentive}} & C_{\text{inattentive},\text{uncertain}} \\ C_{\text{attentive},\text{inattentive}} & C_{\text{attentive},\text{attentive}} & C_{\text{attentive},\text{uncertain}} \end{pmatrix}$$
* **Coverage Ratio**:
  $$\text{Coverage} = \frac{N_{\text{evaluable}}}{N_{\text{total visible students}}}$$

### 3. Epistemological Guardrails
* **Downward gaze $\neq$ inattention**: A student looking downward while note-taking or reading a handout is marked as active engagement if torso posture is upright ($\ge 75^\circ$) and subtle hand/motion energy is present.
* **Head slumped $\ge 3.0\text{s}$ with low motion $\to$ verifiable behavioral fatigue alert**.
* **Zero Cognitive Telepathy**: Explicitly state in the thesis that video systems measure observable physical alignment with instruction, not internal cognitive comprehension.

---

## 5. Next Smallest Useful Experiment: Concrete Recommendation

### Recommendation: Execute the 3-Clip Individual Annotation Pilot
Instead of spending days downloading 16.5 GB of group-labeled video that cannot solve the individual ground-truth problem:

1. **Take 3 Diverse Clips from the Existing Pilot**:
   - `high/view1.mp4` (Front View)
   - `low/view3.mp4` (Side View)
   - `high/view11.mp4` (Rear / High-Angle View)
2. **Apply Legitimate Individual Ground-Truth Annotations**:
   - For each student track $T_i$, record second-by-second observable ground truth:
     `[0s–10s]: Student #1 = Attentive (Facing board, upright, note-taking)`
     `[4s–9s]:  Student #3 = Inattentive (Head slumped downward on desk, inactive)`
     `[0s–10s]: Student #6 = Uncertain / Distant (Rear row, face unresolvable)`
3. **Run Full Pipeline Benchmark**:
   - Compute exact individual precision, recall, macro F1, and tracking continuity against verified ground truth.
4. **Outcome**: Produces an honest, statistically defensible pilot benchmark for the CSE thesis demonstrating that the pipeline functions end-to-end on individual students in real classrooms before touching any full dataset.

---

## 6. Exact CLI Reproduction Commands

```powershell
# 1. Inspect the dataset identity audit
cat classroom_attention_research/reports/dataset_identity_audit.md

# 2. Inspect the pilot verification metrics and confusion matrix proofs
cat classroom_attention_research/reports/pilot_verification_metrics.json

# 3. View the rendered honest pilot pipeline demo video
# (Shows YOLO11 detection, DeepSORT tracking, and calibrated observable scoring)
start classroom_attention_research/reports/honest_pilot_pipeline_demo.mp4

# 4. Review the minimal individual pilot plan
cat classroom_attention_research/reports/dataset_pilot_plan.md
```

---

## 7. Hard Stop Confirmation

In strict compliance with your instructions:
* **Zero bytes** of the full OUC-CGE dataset have been downloaded.
* **Zero full-scale training** has been initiated.
* All existing pilot data, manifests, checkpoints, and reports have been preserved without overwriting.
* **Execution is halted. Awaiting your explicit review and authorization.**
