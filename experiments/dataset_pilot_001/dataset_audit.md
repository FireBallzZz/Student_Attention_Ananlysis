# Comprehensive Dataset Audit & Schema Verification

**Repository**: `classroom_attention_research`  
**Run Directory**: `experiments/dataset_pilot_001/`  
**Audit Date**: October 27, 2026  
**Auditor**: Senior Computer Vision Research Engineer & Experimental-Audit Specialist  

---

## 1. Candidate Dataset Inventory Matrix

| Candidate Dataset | Modality & File Type | Size on Disk / Download | License & Terms | Ground-Truth Level | Accessibility & Blocker Status | Pilot Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mendeley `smzggbnkd2/3`** | Tabular CSV (38 columns, 6,244 rows) | 1.53 MB (1,609,237 bytes) | **CC BY 4.0** (Open Access) | Individual attention (0=Attentive, 1=Inattentive) | **Direct Public Access** (No login/form) | **Experiment A**: Pre-extracted tabular feature baseline |
| **SCB-Dataset (SCB5)** | Static classroom images + YOLO txt bboxes | Pilot: ~14.2 MB (100 images) Full: ~779 MB | Non-commercial Academic Research | Object bounding boxes for visible physical actions | **Accessible via HTTP range** from HuggingFace mirror | **Experiment B**: Student action detection smoke test |
| **DAiSEE (IIT Mandi)** | 9,068 raw video clips (10s snippets) | ~15 GB compressed | Academic EULA (Requires signed form) | 4 Affective States (Engagement, Boredom, Confusion, Frustration) | <span style="color:red; font-weight:bold;">BLOCKED</span>: Requires institutional registration form; 15 GB exceeds pilot limits | Excluded from pilot experiments |
| **OUC-CGE Pilot** | 24 multi-camera continuous video clips (30 FPS) | 41.8 MB (pilot subset) Full: ~16.5 GB | **CC BY 4.0** (Open Access) | Clip-level cohort engagement (High, Medium, Low) | **Locally Available** in `data/pilot/` | **Experiment C**: Group engagement audit & session leakage evaluation |

---

## 2. In-Depth Audit: Mendeley Attention Dataset (`smzggbnkd2/3`)

### 2.1 Provenance & Metadata
* **Title**: *Students Attention Detection Dataset*, Version 3
* **DOI**: `10.17632/smzggbnkd2.3`
* **Authors**: Muhammad Kamal Hossen (CUET) & Mohammad Shorif Uddin (Jahangirnagar University), published August 27, 2026.
* **Download Source**: Mendeley Data public file API (`attention_detection_dataset_v3.csv`).
* **Cryptographic Hashes**:
  - File Size: `1,609,237 bytes`
  - MD5: `1280c448eed54e264faf123e6add2e15`
  - SHA-256: `b577cb262d35f8b4b31db3bcb4132e82ed09aca0514f430178f3d3959154e685`

### 2.2 Schema & Feature Categories (38 Columns)
The dataset comprises 37 input features pre-extracted from webcam video feeds during virtual learning sessions and 1 binary target column:
1. **Face Geometry & Detection (7 features)**: `face_present`, `no_of_face`, `face_x`, `face_y`, `face_w`, `face_h`, `face_conf`.
2. **Facial Landmarks (8 features)**: `left_eye_x`, `left_eye_y`, `right_eye_x`, `right_eye_y`, `nose_tip_x`, `nose_tip_y`, `mouth_x`, `mouth_y`.
3. **Hand Activity & Manipulation (6 features)**: `hand_count`, `left_hand_x`, `left_hand_y`, `right_hand_x`, `right_hand_y`, `hand_obj_interaction`.
4. **Head Pose Orientation (4 features)**: `head_pose` (categorical: `forward`, `right`, `left`, `down`), `head_pitch`, `head_yaw`, `head_roll`.
5. **Mobile Device Detection (4 features)**: `phone_present`, `phone_loc_x`, `phone_loc_y`, `phone_conf`.
6. **Gaze & Screen Tracking (4 features)**: `gaze_on_screen`, `gaze_direction` (categorical), `gazePoint_x`, `gazePoint_y`.
7. **Pupil Center Coordinates (4 features)**: `pupil_left_x`, `pupil_left_y`, `pupil_right_x`, `pupil_right_y`.
8. **Target Label (1 column)**: `label` ($0 = \text{Attentive}$, $1 = \text{Inattentive}$).

### 2.3 Class Balance, Missingness, & Target Leakage
* **Class Distribution (Full Dataset, N=6,244)**:
  - Class 0 (Attentive): 3,244 instances (51.95%)
  - Class 1 (Inattentive): 3,000 instances (48.05%)
* **Missing Values**:
  - `head_pose`: 259 missing values ($4.15\%$). Handled via `SimpleImputer(strategy="constant", fill_value="missing")`.
  - `gaze_direction`: 1,179 missing values ($18.88\%$). Handled via `SimpleImputer(strategy="constant", fill_value="missing")`.
  - All other 36 columns have exactly 0 missing values.
* **Exact Duplicate Rows**: Exactly 1 duplicate row detected and isolated.
* **Target Leakage Audit**:
  - `gaze_on_screen` exhibits a correlation of $-0.664$ with `label`. When gaze is oriented to the laptop screen, the student is predominantly attentive.
  - No feature exhibits a deterministic correlation of $\pm 1.0$, confirming the absence of direct target leakage.
* **Grouping Metadata**: **Zero subject IDs, user IDs, or session timestamps are provided.** The data were anonymized and flattened by the upstream authors. Group-based splitting was therefore technically impossible; a deterministic stratified split (Seed 42) was enforced.

### 2.4 Pilot Partition Manifest (N=1,500)
To satisfy the resource constraint (target: 1,000–2,000 rows):
* **Pilot Subset**: Exactly 1,500 samples (779 Attentive [51.93%], 721 Inattentive [48.07%]).
* **Train Split (70%)**: 1,050 samples (545 Attentive, 505 Inattentive).
* **Held-Out Test Split (30%)**: 450 samples (234 Attentive, 216 Inattentive).

---

## 3. In-Depth Audit: Student Classroom Behavior Dataset (SCB-Dataset)

### 3.1 Provenance & Distribution Setup
* **Repository**: `Whiffe/SCB-dataset` (GitHub) & `wintonYF/SCB-Dataset` (HuggingFace).
* **Reference**: Yang et al., *"SCB-dataset: Student Classroom Behavior Dataset"*, arXiv:2304.02488 / Springer 2023.
* **Licensing**: Strictly non-commercial academic research only. Commercial use requires explicit authorization.
* **Data Format**: Static JPEG images ($1280 \times 720$ or $1920 \times 1080$) paired with YOLO normalized bounding box text files (`<class_id> <x_center> <y_center> <width> <height>`).

### 3.2 Action Class Taxonomy
* Class 0: `hand-raising` (Active student raising arm/hand to volunteer or ask a question).
* Class 1: `read` (Eyes and head directed downward at book or worksheet).
* Class 2: `write` (Hand with writing utensil actively engaging on desk paper/notebook).

### 3.3 Pilot Extraction & Split Integrity (N=100)
Because SCB is distributed in large monolithic zip archives (779 MB for Handrise-Read-Write), we executed **seekable HTTP range downloads** to extract a targeted pilot sample:
* **Train Images**: 70 images (extracted from distinct session recording prefixes including `3005`, `3000`, `0089`, `3004`).
* **Val Images**: 30 images (extracted from non-overlapping held-out session recording prefixes including `3001`, `10`, `3002`).
* **Session Leakage Prevention**: Verified that recording ID prefixes between train and val partitions have zero overlap, preventing near-duplicate video frame contamination.
* **Annotation Integrity**: 100% of selected images have non-empty, syntactically valid YOLO label files.

---

## 4. In-Depth Audit: Local OUC-CGE Pilot Dataset

### 4.1 Checksum & Integrity Audit
The 24 local pilot clips in `data/pilot/` were verified against `data/manifests/pilot_manifest.json`:
* **Total Clips on Disk**: 24 clips ($41.85\text{ MB}$).
* **Corrupted / Frozen Clips Discovered**: 2 clips (`low/view7.mp4` and `low/view8.mp4` freeze on frame 1 with zero pixel motion). **Excluded.**
* **Duplicate Footage Discovered**: `low/view2.mp4` is byte-identical (MD5: `07cac49efbcdbd45e2d2bb905ff22eeb`) to `low/view1.mp4`. **Excluded.**
* **Clean Dynamic Evaluation Subset**: Exactly 21 clips (12 High Engagement, 9 Low Engagement).

### 4.2 Critical Scientific Findings on OUC-CGE
1. **Ecological Fallacy Hazard**:
   - Upstream ground-truth labels (`High`, `Medium`, `Low`) are **clip-level cohort engagement atmospheres**.
   - The dataset contains **zero individual student attention ground-truth annotations**.
   - Projecting group engagement onto individual student tracks commits an ecological fallacy and is barred from thesis benchmarking.
2. **Multi-View Synchronization Leakage**:
   - The 21 dynamic clips stem from only **two physical classroom sessions** recorded by 12 synchronized fixed cameras simultaneously.
   - Cross-validation across views inside the same session evaluates multi-camera view invariance, **NOT generalization to unseen classrooms**.
