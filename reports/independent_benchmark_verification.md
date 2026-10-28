# Independent Scientific Audit & Verification Report: Clip 4 Benchmark

**Audit Execution Date**: October 28, 2026  
**Auditor**: Independent Senior Computer Vision & Machine Learning Verification Specialist  
**Evaluated Artifacts**:
- Input Video: [`data/clip 4.mp4`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/data/clip%204.mp4) (832×464 @ 29.97 FPS, 300 frames, 10.01s)
- Ground Truth File: [`data/annotate for clip 4/annotations.xml`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/data/annotate%20for%20clip%204/annotations.xml) (CVAT XML 1.1, 6,300 boxes across 21 tracks)
- Evaluated Checkpoint: [`checkpoints/final_attention_model.joblib`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/checkpoints/final_attention_model.joblib)
- Reported Benchmark Report: [`reports/clip_4_accuracy_evaluation.md`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/reports/clip_4_accuracy_evaluation.md)
- Machine-Readable Audit Metrics: [`reports/independent_benchmark_metrics.json`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/reports/independent_benchmark_metrics.json)

---

## 1. Executive Summary & Verification Verdict

An exhaustive, non-destructive audit was performed to independently verify the reported accuracy on `data/clip 4.mp4`. The mathematical computation of the reported metrics was fully reproduced, but the audit reveals **severe methodological flaws and fundamental data leakage** that prevent these metrics from representing a valid out-of-sample benchmark.

```
+------------------------------------------------------------------------------------------------------------------+
|                                            AUDIT VERIFICATION VERDICT                                            |
+----------------------------------+------------------+--------------------+---------------------------------------+
| Claim / Dimension                | Reported Value   | Independently Recomputed| Audit Verification Status        |
+----------------------------------+------------------+--------------------+---------------------------------------+
| 3-Class Overall Accuracy         | 93.30%           | 93.30% (5,545 / 5,943)| Mathematically Reproduced (Leaked)|
| 3-Class Balanced Accuracy        | 94.54%           | 94.54%             | Mathematically Reproduced (Leaked)    |
| Multi-Class Macro F1             | 0.922            | 0.9217 (0.922)     | Mathematically Reproduced (Leaked)    |
| Binary Accuracy                  | 93.61%           | 93.61%             | Mathematically Reproduced (Leaked)    |
| Student Discovery Rate           | 21 / 21 (100.0%) | 21 / 21 (At least once)| Semantically Misleading (Coverage < 100%)|
| Spatial IoU Recall (@ IoU 0.30)  | 91.14%           | 91.14% (5,742 / 6,300)| Mathematically Reproduced         |
| Spatial IoU Precision (@ IoU 0.30)| 83.54%          | 83.54% (5,742 / 6,873)| Mathematically Reproduced         |
| Mean Bounding Box IoU            | 0.631            | 0.6306 (0.631)     | Mathematically Reproduced             |
| Out-of-Sample Generalization     | Claimed Robust   | INVALID / FAILED   | SEVERE LEAKAGE (Trained on Test Data) |
+----------------------------------+------------------+--------------------+---------------------------------------+
```

> [!CAUTION]
> **CRITICAL SCIENTIFIC FINDING**:
> The model checkpoint `final_attention_model.joblib` was **directly fit on feature vectors extracted from `data/clip 4.mp4`** (`src/models/train_final_model.py`, Lines 31–35, 126–133). 
> Evaluating this model on `data/clip 4.mp4` constitutes **train-set re-substitution** (testing on training data). Therefore, the 93.3% accuracy represents **training memory / memorization capacity**, NOT generalizable out-of-sample attention detection accuracy.

---

## 2. Checkpoint Inspection (`checkpoints/final_attention_model.joblib`)

Loading and inspecting the binary dictionary in `final_attention_model.joblib`:
* **Exact Model Architecture**: `sklearn.ensemble.RandomForestClassifier`
  - `n_estimators`: 120
  - `max_depth`: 12
  - `class_weight`: `"balanced"`
  - `criterion`: `"gini"`
  - `random_state`: 42
* **Classes Saved**: `['High', 'Low', 'Moderate']`
* **Feature Schema (16 Columns)**:
  `['head_pitch_deg', 'head_yaw_deg', 'head_roll_deg', 'head_pose_confidence', 'gaze_horizontal', 'gaze_vertical', 'torso_angle_deg', 'shoulder_tilt_deg', 'movement_energy', 'bbox_area_ratio', 'detection_confidence', 'missing_face', 'missing_head_pose', 'missing_gaze', 'missing_torso', 'is_distant']`
* **Saved Preprocessing**: **None.** No `StandardScaler`, `MinMaxScaler`, or pipeline is stored within the checkpoint. The raw unnormalized 16D feature vector is fed directly to the Random Forest trees.
* **Checkpoint Sample Count**: `trained_samples: 5732`
* **Internal Checkpoint Metrics**:
  - `5-Fold CV Accuracy`: 89.55%
  - `5-Fold CV Balanced Accuracy`: 88.10%

---

## 3. Training & Validation Split Manifest Audit (Data Leakage Verification)

Inspection of `src/models/train_final_model.py` reveals the exact dataset origin of the checkpoint:
```python
# Lines 31-35, 126-133 of src/models/train_final_model.py
xml_path = Path("data/annotate for clip 4/annotations.xml")
video_path = Path("data/clip 4.mp4")
# ... extracts 5,732 observations from clip 4 ...
final_model = RandomForestClassifier(n_estimators=120, max_depth=12, class_weight="balanced", random_state=42)
final_model.fit(X, y)
joblib.dump(artifact, checkpoint_out)
```

### Audit Findings on Data Split:
1. **Zero Out-of-Distribution Split**: There is **no separate train/test split manifest** for the Clip 4 benchmark. The entire clip was extracted, used to train `final_attention_model.joblib`, and then re-evaluated with the exact same script.
2. **Temporal Autocorrelation Leakage**: The internal cross-validation recorded in the checkpoint used `StratifiedKFold(n_splits=5, shuffle=True)` across consecutive video frames. Shuffling 30 FPS video frames places frame $t$ in the training fold and frame $t+1$ (33 milliseconds later) in the validation fold. Because human posture and gaze do not change significantly in 33ms, this produces near-total temporal frame leakage.
3. **Absence of Independent Test Video**: The benchmark contains no evaluation on unseen students, an unseen camera angle, or an independent recording session.

---

## 4. Ground Truth Audit (`data/annotate for clip 4/annotations.xml`)

Inspection of the raw XML structure:
* **Tool & Provenance**: Exported from **CVAT.ai (Computer Vision Annotation Tool)** v1.1.
* **CVAT Metadata**:
  - Task ID: `4561675`, Segment: `4558907`
  - Created: `2026-10-07 15:17:08 UTC`, Exported: `2026-10-07 18:07:33 UTC`
  - Owner: `FireBallzZz` (`shrajon10@gmail.com`), Assignee: `junaid21` (`junaidalrashid555@gmail.com`)
  - Interpolation Mode: Linear interpolation between keyframes.
* **Tracks & Boxes**:
  - Exactly **21 tracks** (Track IDs `0` through `20`), representing 21 seated students.
  - Exactly **300 boxes per track** (one box per video frame from frame 0 to 299).
  - Total bounding boxes: $21 \times 300 = \mathbf{6,300}\text{ boxes}$.
* **Label Integrity**:
  - Every bounding box contains an explicit `<attribute name="Attention">` tag with one of three values: `High`, `Moderate`, or `Low`.
  - XML distribution across all 6,300 boxes:
    - **High**: 4,100 boxes (65.08%)
    - **Moderate**: 1,082 boxes (17.17%)
    - **Low**: 1,118 boxes (17.75%)
* **Alignment to Timestamps**:
  Boxes are indexed by frame integer (`frame="0"` through `frame="299"`). Timestamps are derived via $t = \text{frame} / 29.97\text{ seconds}$.

---

## 5. Recomputation of Confusion Matrix & Accuracy Metrics

Executing independent evaluation with [`scratch/audit_standalone_clip4.py`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/scratch/audit_standalone_clip4.py) produced the following mathematical metrics:

### 3-Class Confusion Matrix (Evaluated Support: $N = 5,943$)
$$\begin{array}{r|ccc|c|c}
\textbf{Ground Truth} \;\backslash\; \textbf{Predicted} & \textbf{High} & \textbf{Moderate} & \textbf{Low} & \textbf{Total Support} & \textbf{Recall} \\
\hline
\textbf{High} & \mathbf{3,477} & 237 & 74 & 3,788 & \mathbf{91.79\%} \\
\textbf{Moderate} & 42 & \mathbf{987} & 14 & 1,043 & \mathbf{94.63\%} \\
\textbf{Low} & 27 & 4 & \mathbf{1,081} & 1,112 & \mathbf{97.21\%} \\
\hline
\textbf{Total Predicted} & 3,546 & 1,228 & 1,169 & \mathbf{5,943} & \text{Accuracy: } \mathbf{93.30\%} \\
\textbf{Precision} & \mathbf{98.05\%} & \mathbf{80.37\%} & \mathbf{92.47\%} & &
\end{array}$$

### Per-Class Detailed Performance
* **High Attention**: Precision = **98.05%**, Recall = **91.79%**, F1 = **0.9482** (Support: 3,788)
* **Moderate Attention**: Precision = **80.37%**, Recall = **94.63%**, F1 = **0.8692** (Support: 1,043)
* **Low Attention**: Precision = **92.47%**, Recall = **97.21%**, F1 = **0.9478** (Support: 1,112)
* **Macro F1**: $\frac{0.9482 + 0.8692 + 0.9478}{3} = \mathbf{0.9217}$
* **Balanced Accuracy**: $\frac{91.79\% + 94.63\% + 97.21\%}{3} = \mathbf{94.54\%}$

### Binary Performance (High Focus vs. Disengaged)
* **Binary Accuracy**: **93.61%**
* **Binary Balanced Accuracy**: **94.29%**
* **Binary F1-Score**: **0.9482**
* **Binary Confusion Matrix**:
  - High (Attentive): 3,477 True Positives, 311 False Negatives
  - Low (Disengaged): 2,086 True Negatives, 69 False Positives

---

## 6. Detection Metrics, IoU Thresholds, and Matching Methodology

### Matching Algorithm
In each frame $t$:
1. An IoU cost matrix between all ground-truth boxes $G_t$ and predicted boxes $P_t$ is constructed.
2. A **greedy bipartite assignment** repeatedly selects the maximum IoU pair $(g_i, p_j)$.
3. Pairs with $\text{IoU} < 0.20$ are rejected.
4. Matched rows and columns are zeroed out until no candidate pairs exceed $\text{IoU} \ge 0.20$.

### Spatial Detection Performance
* **Mean Matched IoU**: **0.6306**
* **Total Predicted Boxes**: 6,873 across 300 frames (mean: 22.9 boxes/frame).
* **True Positives (@ IoU 0.30)**: 5,742
  - **Precision (@ IoU 0.30)**: $\frac{5,742}{6,873} = \mathbf{83.54\%}$
  - **Recall (@ IoU 0.30)**: $\frac{5,742}{6,300} = \mathbf{91.14\%}$
  - **Detection F1 (@ IoU 0.30)**: $\mathbf{87.18\%}$
* **True Positives (@ IoU 0.50)**: 4,830
  - **Precision (@ IoU 0.50)**: $\frac{4,830}{6,873} = \mathbf{70.27\%}$
  - **Recall (@ IoU 0.50)**: $\frac{4,830}{6,300} = \mathbf{76.67\%}$
  - **Detection F1 (@ IoU 0.50)**: $\mathbf{73.33\%}$
* **False Positive Predictions**: **930 detections** had no matching ground-truth student box (often background desk artifacts or split duplicate bounding boxes).

---

## 7. Reconciliation: Why 5,943 Observations Differ from 6,300 XML Boxes

A key audit question is why the evaluation evaluated **5,943 instances** instead of the full **6,300 ground-truth boxes**.

$$\text{Total XML Ground Truth Boxes (6,300)} = \underbrace{\text{Matched Instances (5,943)}}_{\text{Evaluated for Attention}} + \underbrace{\text{Missed GT Boxes (357)}}_{\text{Dropped from Attention Evaluation}}$$

### Audit Finding on Evaluation Attrition Bias:
* Exactly **357 ground-truth student boxes** were **completely missed** by YOLO11-Pose (or failed the $\text{IoU} \ge 0.20$ threshold).
* **The evaluation methodology silently dropped these 357 missed instances from the attention classification accuracy.**
* **Scientific Impact**: Students who were occluded or hard to detect were excluded from the classification metrics. If missed students are treated as detection failures, the end-to-end multi-student recognition accuracy drops proportionally:
  $$\text{End-to-End Recall} = \frac{5,545\text{ correct}}{6,300\text{ total ground truth}} = \mathbf{88.02\%}$$

---

## 8. Verification of "21 / 21 Student Discovery Rate"

The reported claim that *"21 / 21 (100.0%) of seated students were discovered and tracked"* was audited against frame-by-frame temporal tracking continuity.

### Discovery Semantic Reality:
* **"Discovered" means detected at least once.** All 21 ground-truth tracks had $\ge 1$ matching detection over the 10-second clip.
* **It does NOT mean continuous 100% temporal tracking.**

### Per-Student Coverage Breakdown:
* **18 of 21 students** were tracked across **100.0% of frames** (300/300 frames).
* **3 students suffered significant occlusion and tracking dropouts**:
  - **Student #15**: Matched in only **75 of 300 frames (25.0% coverage)**. Missed in 225 frames!
  - **Student #18**: Matched in **208 of 300 frames (69.33% coverage)**. Missed in 92 frames!
  - **Student #16**: Matched in **260 of 300 frames (86.67% coverage)**. Missed in 40 frames!
  - Sum of missed frames: $225 + 92 + 40 = \mathbf{357}\text{ frames}$ (accounting for 100% of the discrepancy to 6,300).
* **Track ID Fragmentation & Identity Switches**:
  - Across the 21 ground truth students, **7 identity switches** occurred:
    - Student #6 was assigned 3 different model track IDs (`#16`, `#20`, `#28`) — 2 switches.
    - Student #13 was assigned 3 different model track IDs (`#12`, `#26`, `#30`) — 2 switches.
    - Student #16 was assigned 3 different model track IDs (`#21`, `#32`, `#34`) — 2 switches.
    - Student #15 was assigned 2 different model track IDs (`#31`, `#33`) — 1 switch.
  - The remaining 17 students maintained clean, unbroken identity tracks (0 switches).

---

## 9. Comprehensive Data & Methodological Leakage Catalog

| Leakage Category | Status | Exact Evidence & Mechanism | Scientific Severity |
| :--- | :---: | :--- | :---: |
| **Train-Test Re-substitution** | **CONFIRMED** | `train_final_model.py` fit the Random Forest on 5,732 frames extracted directly from `data/clip 4.mp4`. The benchmark then tested on `data/clip 4.mp4`. | **FATAL** |
| **Temporal Autocorrelation** | **CONFIRMED** | 5-fold cross-validation shuffled consecutive 30 FPS video frames (`StratifiedKFold(shuffle=True)`). Frames $t$ and $t+1$ shared identical physical states. | **HIGH** |
| **Attrition / Exclusion Bias** | **CONFIRMED** | 357 missed ground-truth instances (5.7%) were silently dropped from the classification denominator rather than penalized. | **MODERATE** |
| **Hardcoded Path Failure** | **CONFIRMED** | `AttentionScorer.__init__` hardcodes `classroom_attention_research/checkpoints/...`. When executed from the repo root, `self.model` evaluates to `None`, silently reverting to an uncalibrated heuristic index (63.5% accuracy) unless path is passed explicitly. | **ENGINEERING** |

---

## 10. Distinguishing Verified Facts vs. Unverified Claims

### Verified Facts (Empirically Confirmed by Audit)
1. [`annotations.xml`](file:///c:/Users/User/Downloads/Model%20v2/classroom_attention_research/data/annotate%20for%20clip%204/annotations.xml) contains genuine, fine-grained individual student bounding boxes and 3-level attention attributes created in CVAT by project contributors.
2. The mathematical calculation of the confusion matrix, 93.3% accuracy, 94.5% balanced accuracy, and spatial IoU in `reports/clip_4_accuracy_evaluation.md` was reproduced.
3. The detector achieves 91.14% recall at IoU 0.30 and 18 of 21 students have 100% continuous temporal coverage.

### Unverified / Scientifically Invalid Claims
1. **"Production-Grade Generalization"**: **CANNOT BE CLAIMED.** The evaluation was conducted on the model's training data. True generalization to new classrooms, unseen students, or novel camera angles remains unproven until evaluated on held-out footage (e.g., Clip 11).
2. **"100% Student Discovery and Tracking"**: **MISLEADING.** While all students were seen at least once, Student #15 was missed in 75% of frames, and 7 track identity switches occurred.

---

## 11. Concrete Recommendations for Thesis Readiness

1. **Establish a Proper Train / Validation / Test Split Across Independent Clips**:
   - Train on `Clip 4` $\to$ Test on `Clip 11` (or vice versa).
   - Alternatively, partition Clip 4 temporally with an embargo gap: Train on frames 0–200, evaluate on frames 220–300.
2. **Fix Default Checkpoint Resolution in `AttentionScorer`**:
   - Update `AttentionScorer.__init__` to dynamically resolve paths using `Path(__file__).resolve().parent.parent.parent / "checkpoints"` so it never fails silently when run from different working directories.
3. **Report Combined Detection + Classification (End-to-End Metric)**:
   - Penalize missed student boxes in the primary metric (e.g., reporting end-to-end recall of 88.02% alongside localized classification accuracy).
