# Small-Scale Dataset Pilot Training and Honest Results Audit Report

**Date**: October 27, 2026  
**Auditor**: Senior Computer Vision Research Engineer, ML Scientist & Experimental-Audit Specialist  
**Experiment ID**: `dataset_pilot_001`  
**Repository**: `c:\Users\User\Downloads\Model v2\classroom_attention_research`  
**Artifacts Location**: [`experiments/dataset_pilot_001/`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/experiments/dataset_pilot_001/)  

---

## 1. Executive Summary & Comparative Candidate Decision Matrix

To establish an evidence-based foundation for student attention analysis without premature full downloads or uncontrolled compute expenditure, three independent pilot experiments were conducted across four candidate datasets.

```
+--------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                      COMPARATIVE PILOT EVALUATION MATRIX                                                         |
+-------------------+---------------+--------------------+----------------------+-------------+-------------------+----------------------------+
| Category & Dataset| Data Type     | GT Level           | Experimental Task    | Sample Size | Split Strategy    | Key Performance Metrics   |
+-------------------+---------------+--------------------+----------------------+-------------+-------------------+----------------------------+
| A. Mendeley v3    | Tabular CSV   | Individual Binary  | Tabular Attention    | 1,500 rows  | Stratified Train/ | Accuracy: 92.67%           |
|    (smzggbnkd2/3) | (37 features) | (0: Att, 1: Inatt) | Classification       | (Pilot)     | Test (70% / 30%)  | Balanced Acc: 92.59%       |
|                   |               |                    |                      |             |                   | Macro F1: 0.9264           |
+-------------------+---------------+--------------------+----------------------+-------------+-------------------+----------------------------+
| B. SCB-Dataset    | Static JPEG   | Bounding Boxes     | Student Physical     | 100 images  | Source Recording  | Precision: 0.0642          |
|    (SCB5 Pilot)   | Images        | (3 Action Classes) | Action Detection     | (Pilot)     | Prefix Isolation  | Recall: 0.2328             |
|                   |               |                    |                      |             | (70 Train, 30 Val)| mAP@0.5: 0.0368            |
+-------------------+---------------+--------------------+----------------------+-------------+-------------------+----------------------------+
| C. OUC-CGE        | Multi-Camera  | Clip Atmosphere    | Group Engagement     | 21 clips    | Leave-One-Out     | Accuracy: 95.24%           |
|    (Pilot Subset) | Video (30fps) | (High, Med, Low)   | Classification       | (Dynamic)   | (LOO-CV)          | Balanced Acc: 95.83%       |
|                   |               |                    |                      |             |                   | Macro F1: 0.9519           |
+-------------------+---------------+--------------------+----------------------+-------------+-------------------+----------------------------+
| D. Physical Video | Multi-Student | Individual         | Video Student        | 0 clips     | N/A               | NOT APPLICABLE             |
|    Individual GT  | Classroom Vid | Attention States   | Attention Tracking   |             | (No Public Video  | (No off-the-shelf public   |
|                   |               | over Time          | in Physical Space    |             | GT Exists)        | video dataset exists)      |
+-------------------+---------------+--------------------+----------------------+-------------+-------------------+----------------------------+
```

### Limitations & Role in Master's Thesis

| Dataset | Main Technical & Epistemic Limitation | Evaluated Result Label | Strategic Recommendation for Thesis |
| :--- | :--- | :---: | :--- |
| **Mendeley v3** | **Zero raw video or images.** Derived from virtual online laptop webcams. Cannot evaluate YOLO detection, DeepSORT tracking, desk occlusions, or distance gating. | **Proper Validation Result** (On Tabular Features) | **Retain as Tabular Feature Baseline**: Validates multimodal fusion algorithms on pre-extracted feature vectors, but cannot benchmark video tracking. |
| **SCB-Dataset** | **Static images only; zero continuous video tracking sequences.** Labels reflect visible physical actions (`hand-raising`, `reading`, `writing`), NOT cognitive attention states. | **Pipeline Smoke Test** (Small Pilot Subset) | **Retain for Action Detector Pretraining / Verification**: Useful for calibrating YOLO behavior detection heads, but cannot evaluate multi-object temporal tracking. |
| **OUC-CGE** | **Zero individual attention ground truth.** Multi-camera views within the same session exhibit severe synchronization leakage. | **Small Pilot Result** (Group-Level Atmosphere) | **Retain as Video Pipeline Engine Only**: Perfect for validating YOLO11-Pose detection, DeepSORT identity continuity, and spatial distance gating; never use group labels as student ground truth. |
| **DAiSEE** | **Gated behind institutional EULA form; 15 GB archive exceeds pilot limits.** Single-user laptop e-learning webcam domain. | <span style="color:red; font-weight:bold;">BLOCKED / EXCLUDED</span> | **Reject**: Substantial domain gap from multi-student physical classrooms and restricted distribution terms. |

> [!IMPORTANT]
> **NO COLLAPSED METRICS**: In accordance with rigorous scientific standards, metrics from Categories A, B, and C are **never aggregated into a single accuracy number**. They evaluate fundamentally distinct tasks: Tabular Feature Classification, Static Object Detection, and Cohort Atmosphere Recognition.

---

## 2. Resource Audit & Data Integrity

### 2.1 Hardware and Compute Profile
* **Host Platform**: Windows 11 Professional (AMD64)
* **Python Runtime**: Python 3.14.4
* **Acceleration**: NVIDIA GeForce RTX 3060 (12.00 GB GDDR6 VRAM, CUDA Capability 8.6)
* **PyTorch Environment**: PyTorch `2.14.1+cu126`, Torchvision `0.29.1+cu126`, Ultralytics `8.4.182`
* **Disk Status**: 930.40 GB Total, 398.86 GB Free Available
* **Peak Training GPU Memory**: $1,146.48\text{ MB}$ (9.3% of available VRAM during YOLO11n fine-tuning)

### 2.2 Downloaded Files & Cryptographic Verification
Strict pilot resource limits were enforced. Zero full datasets were downloaded. Only two minimal, documented subsets were fetched:

1. **`attention_detection_dataset_v3.csv` (Mendeley Data)**:
   - File Size: $1,609,237\text{ bytes}$ ($1.53\text{ MB}$)
   - MD5: `1280c448eed54e264faf123e6add2e15`
   - SHA-256: `b577cb262d35f8b4b31db3bcb4132e82ed09aca0514f430178f3d3959154e685`
   - License: Creative Commons Attribution 4.0 International (CC BY 4.0)
2. **SCB-Dataset Pilot (100 image-label pairs via seekable HTTP range)**:
   - Download Volume: $12.80\text{ MB}$ ($13,421,772\text{ bytes}$)
   - Source Archive: `wintonYF/SCB-Dataset/SCB5-Handrise-Read-write-2024-9-17.zip`
   - Image Count: 70 train images, 30 validation images (with 100 matching `.txt` YOLO labels)
   - License: Non-commercial Academic Research

---

## 3. Experiment A: Tabular Attention Classification (Mendeley `smzggbnkd2/3`)

### 3.1 Experimental Methodology & Pipeline
* **Target Task**: Binary classification ($0 = \text{Attentive}$, $1 = \text{Inattentive}$).
* **Pilot Subset**: 1,500 rows stratified from the 6,244-record dataset (779 Attentive, 721 Inattentive).
* **Partitioning**: 70% Train ($N=1,050$), 30% Held-Out Test ($N=450$), stratified by target label (Random Seed 42).
* **Preprocessing Pipeline** (Fitted strictly on train partition):
  - Numeric features ($N=35$): `SimpleImputer(strategy="median")` $\to$ `StandardScaler()`.
  - Categorical features (`head_pose`, `gaze_direction`): `SimpleImputer(strategy="constant", fill_value="missing")` $\to$ `OneHotEncoder(handle_unknown="ignore")`.
* **Grouping Metadata Audit**: The dataset provides zero student or session IDs. Group-based partitioning was structurally impossible.

### 3.2 Quantitative Results

```
+-------------------------------------------------------------------------------------------------------+
|                                 EXPERIMENT A TEST SET METRICS (N=450)                                 |
+------------------------+----------+----------+----------+--------------------+--------------------+---+
| Model Architecture     | Accuracy | Bal. Acc | Macro F1 | Attentive (0) F1   | Inattentive (1) F1 | Sec|
+------------------------+----------+----------+----------+--------------------+--------------------+---+
| Dummy (Most Frequent)  | 0.5200   | 0.5000   | 0.3421   | 0.6842 (R=1.0000)  | 0.0000 (R=0.0000)  |0.01|
| Logistic Regression    | 0.9244   | 0.9234   | 0.9242   | 0.9289 (P=0.9098)  | 0.9194 (P=0.9417)  |0.02|
| Random Forest (Best)   | 0.9267   | 0.9259   | 0.9264   | 0.9305 (P=0.9170)  | 0.9224 (P=0.9378)  |0.10|
| HistGradientBoosting   | 0.9200   | 0.9192   | 0.9197   | 0.9244 (P=0.9091)  | 0.9146 (P=0.9327)  |1.67|
+------------------------+----------+----------+----------+--------------------+--------------------+---+
```

* **5-Fold Stratified Cross-Validation on Train Set**:
  - Logistic Regression: $92.48\% \pm 1.43\%$
  - Random Forest: $91.71\% \pm 1.82\%$
  - HistGradientBoosting: $91.33\% \pm 2.16\%$
* **Confusion Matrix for Random Forest (Held-out Test Set, N=450)**:
  $$\begin{pmatrix} C_{\text{Att}\to\text{Att}} & C_{\text{Att}\to\text{Inatt}} \\ C_{\text{Inatt}\to\text{Att}} & C_{\text{Inatt}\to\text{Inatt}} \end{pmatrix} = \begin{pmatrix} 221 & 13 \\ 20 & 196 \end{pmatrix}$$
* **Diagnostic Visualization**: [`confusion_matrix_tabular.png`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/experiments/dataset_pilot_001/visualizations/confusion_matrix_tabular.png)

### 3.3 Epistemic Takeaway
Mendeley v3 demonstrates that **when clean visual features are already extracted**, classical ML models (Random Forest, Logistic Regression) readily achieve $>92\%$ macro F1. However, because this dataset contains **zero raw video**, it bypasses the entire computer vision challenge: multi-person detection, DeepSORT tracking, desk occlusion, low-resolution faces ($<24\text{ px}$), and real-time inference latency in physical classrooms.

---

## 4. Experiment B: Student Action Detection Smoke Test (SCB-Dataset)

### 4.1 Experimental Methodology
* **Target Task**: Student physical behavior bounding box detection.
* **Target Classes**: `0: hand-raising`, `1: read`, `2: write`.
* **Pilot Subset**: 100 images (70 train, 30 validation) with non-overlapping recording session prefixes.
* **Architecture**: Ultralytics YOLO11n (`yolo11n.pt` pretrained baseline).
* **Settings**: 5 epochs, batch size 8, image resolution 640, early stopping patience 3.
* **Execution Time**: 39.80 seconds on NVIDIA RTX 3060 (Peak VRAM: 1,146 MB).

### 4.2 Quantitative Smoke Test Results
* **Overall Metrics (Validation Set: 30 images, 146 ground-truth instances)**:
  - Precision ($\text{P}$): $0.0642$
  - Recall ($\text{R}$): $0.2328$
  - $\text{mAP@0.5}$: $0.0368$
  - $\text{mAP@0.5:0.95}$: $0.0169$
* **Per-Class Metrics**:
  - `hand-raising` (126 instances): Precision $0.0982$, Recall $0.0079$, $\text{mAP@0.5} = 0.0279$
  - `read` (14 instances): Precision $0.0621$, Recall $0.3571$, $\text{mAP@0.5} = 0.0523$
  - `write` (6 instances): Precision $0.0322$, Recall $0.3333$, $\text{mAP@0.5} = 0.0301$
* **Diagnostic Visualization**: [`scb_gt_vs_pred.png`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/experiments/dataset_pilot_001/visualizations/scb_gt_vs_pred.png) shows ground-truth boxes (green) versus fine-tuned YOLO11n predictions (blue).

### 4.3 Honest Scientific Assessment
This experiment is labeled strictly as a **Pipeline Smoke Test**. Training for only 5 epochs on 70 images proves end-to-end compatibility with YOLO11 and verifies dataset schema integrity, but is insufficient for production-level bounding box regression. Crucially, **detecting that a student is reading or writing is not proof of mental attention or inattention**—it captures physical motor behaviors only.

---

## 5. Experiment C: Local OUC-CGE Pilot Audit & Leakage Analysis

### 5.1 Dataset Audit & Exclusions
* **Total Local Clips**: 24 clips ($41.85\text{ MB}$) in `data/pilot/`.
* **Confirmed Corrupt / Frozen Clips**: 2 clips (`low/view7.mp4` and `low/view8.mp4` contain static frozen frames). **Excluded.**
* **Confirmed Duplicate Recording**: `low/view2.mp4` is a bitwise exact duplicate of `low/view1.mp4`. **Excluded.**
* **Dynamic Clean Cohort**: Exactly 21 clips (12 High Engagement, 9 Low Engagement).

### 5.2 Group-Level Classification Performance
Using aggregated student-track features (mean head pose, gaze confidence, movement energy):
* **Logistic Regression (Leave-One-Out CV, N=21)**:
  - Accuracy: $95.24\%$
  - Balanced Accuracy: $95.83\%$
  - Macro F1: $0.9519$
  - Confusion Matrix: $\begin{pmatrix} 9 & 0 \\ 1 & 11 \end{pmatrix}$
* **Random Forest (Leave-One-Out CV, N=21)**:
  - Accuracy: $100.0\%$ (LOO-CV)
* **Diagnostic Visualization**: [`confusion_matrix_ouc_group.png`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/experiments/dataset_pilot_001/visualizations/confusion_matrix_ouc_group.png)

### 5.3 Forensic Multi-View Leakage Finding
> [!CAUTION]
> **MULTI-VIEW SYNCHRONIZATION LEAKAGE**: The 21 dynamic clips stem from only **two physical classroom sessions** recorded simultaneously by 12 synchronized fixed cameras. Cross-validation across views inside the same session evaluates multi-camera view invariance, **NOT generalization to unseen classrooms**. Furthermore, OUC-CGE provides **zero individual student ground truth**. Claiming individual attention accuracy from OUC-CGE commits an ecological fallacy.

---

## 6. Synthesis: What Should Be Developed Next?

Based on the empirical evidence gathered across all three pilot experiments:

1. **Reject Full Downloads of Mendeley and DAiSEE**:
   - Mendeley is pure tabular CSV (no video). Downloading its full versions adds zero computer vision utility.
   - DAiSEE is restricted by EULA, 15 GB in size, and domain-mismatched (single-user laptop webcams).
2. **Prioritize the Defensible Two-Tier Architecture**:
   - **Tier 1 (Continuous Vision Pipeline)**: Use the verified local OUC-CGE pilot footage for technical student detection (YOLO11), multi-object tracking (DeepSORT), and spatial occlusion gating.
   - **Tier 2 (Legitimate Ground Truth)**: Implement the pre-registered **Individual Attention Annotation Protocol** on the 3 selected pilot clips (`high/view1.mp4`, `low/view3.mp4`, `high/view11.mp4`, totaling ~36.5 seconds) once institutional ethics sign-off is secured.
3. **Calibrate Action Heads with SCB-Dataset**:
   - SCB-Dataset can serve as a supplementary pretraining benchmark for student physical action detection (`reading`, `writing`, `hand-raising`), maintaining strict epistemic boundaries between observable physical actions and cognitive attention states.
