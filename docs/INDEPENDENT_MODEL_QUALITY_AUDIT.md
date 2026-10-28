# Independent Model Quality & Benchmark Audit Report

**Audit Execution Date:** October 28, 2026  
**Auditor Role:** Senior Computer Vision Researcher & Machine Learning Systems Auditor  
**Primary Checkpoint Audited:** `checkpoints/final_attention_model.joblib`  
**Evaluation Target Clip:** `data/clip_011.mp4`  
**Reference Training Clip:** `data/clip 4.mp4`  
**Machine-Readable Audit Summary:** `reports/audit/independent_audit_summary.json`  

---

## Executive Summary & Audit Verdict

```
+----------------------------------------------------------------------------------------------------+
|                                      FINAL AUDIT STATUS: BLOCKED                                   |
+----------------------------------------------------------------------------------------------------+
|  Classification benchmarking on data/clip_011.mp4 is BLOCKED because NO individual ground-truth   |
|  attention annotations exist for this recording in any standard format (CVAT XML, JSON, or CSV).  |
|                                                                                                    |
|  The existing reported benchmark of 93.3% accuracy in reports/independent_benchmark_metrics.json   |
|  was confirmed to originate from data/clip 4.mp4, representing train-set re-substitution           |
|  (evaluating on the exact 5,732 observations used to fit the model), NOT independent generalization.|
|                                                                                                    |
|  The non-LLM computer vision ingestion, census tracking, and runtime inference pipeline executes  |
|  reliably on Clip 11 (22 seated students tracked at 67.0% mean attention), but out-of-sample       |
|  classification accuracy claims remain quantitatively unverified until ground truth is established.|
+----------------------------------------------------------------------------------------------------+
```

---

## 1. Confirmed Model and Checkpoint Identity

Inspection of `checkpoints/final_attention_model.joblib`:
- **Model Family / Class:** `sklearn.ensemble.RandomForestClassifier`
- **Ensemble Hyperparameters:**
  - `n_estimators`: 120
  - `max_depth`: 12
  - `class_weight`: `"balanced"`
  - `criterion`: `"gini"`
  - `random_state`: 42
- **Input Dimension:** Exactly 16 features (`n_features_in_ == 16`)
- **Canonical Feature Schema Order:**
  1. `head_pitch_deg` (degrees)
  2. `head_yaw_deg` (degrees)
  3. `head_roll_deg` (degrees)
  4. `head_pose_confidence` ([0.0, 1.0])
  5. `gaze_horizontal` (proxy [-1.0, 1.0])
  6. `gaze_vertical` (proxy [-1.0, 1.0])
  7. `torso_angle_deg` (degrees)
  8. `shoulder_tilt_deg` (degrees)
  9. `movement_energy` (pixels/sec)
  10. `bbox_area_ratio` (ratio)
  11. `detection_confidence` ([0.0, 1.0])
  12. `missing_face` (binary flag: 0.0 or 1.0)
  13. `missing_head_pose` (binary flag: 0.0 or 1.0)
  14. `missing_gaze` (binary flag: 0.0 or 1.0)
  15. `missing_torso` (binary flag: 0.0 or 1.0)
  16. `is_distant` (binary flag: 0.0 or 1.0)
- **Saved Classes & Order:** `['High', 'Low', 'Moderate']`
- **Preprocessing Pipeline Inside Checkpoint:** None. No standard scaler or min-max scaler is bundled inside `final_attention_model.joblib`; raw unscaled 16D feature vectors are evaluated directly by decision trees.
- **Trained Sample Count Recorded in Artifact:** 5,732 observations
- **Internal 5-Fold Cross-Validation Metrics Recorded in Checkpoint:**
  - Overall Accuracy: 89.55%
  - Balanced Accuracy: 88.10%

---

## 2. Verified Training and Evaluation Data Provenance

Inspection of the primary training script `src/models/train_final_model.py` (lines 31–38, 125–145):
```python
# From src/models/train_final_model.py
xml_path = Path("data/annotate for clip 4/annotations.xml")
video_path = Path("data/clip 4.mp4")
# ... extracts 5,732 matched observations from Clip 4 ...
final_model.fit(X, y) # Fit on all 5,732 observations from Clip 4
```

1. **Origin of Training Data:** 100% of the 5,732 training instances used to build `checkpoints/final_attention_model.joblib` originate exclusively from `data/clip 4.mp4` paired with `data/annotate for clip 4/annotations.xml`.
2. **Origin of Reported Benchmark in `reports/independent_benchmark_metrics.json`:**
   - Evaluated video: `data/clip 4.mp4`
   - Evaluated ground-truth: `data/annotate for clip 4/annotations.xml`
   - Matched observations: 5,943 student frames
   - 3-Class Accuracy: 93.30% (5,545 / 5,943 correct)
   - Macro F1: 0.9217
   - **Provenance Finding:** This test was conducted on the **exact same video file** (`clip 4.mp4`) used to fit the model. It is an empirical train-set re-substitution test. It confirms that the Random Forest has high representational capacity (93.3% fit), but provides **no evidence** of generalization to unseen classrooms.

---

## 3. Clip 11 Eligibility Audit

We systematically examined whether `data/clip_011.mp4` was genuinely excluded from all model selection and training phases:

| Check / Phase | Status | Evidence / Verification Method |
| :--- | :---: | :--- |
| **Model Fitting** | **EXCLUDED** | `src/models/train_final_model.py` references only `data/clip 4.mp4`. No feature vectors from `clip_011.mp4` appear in the training matrix. |
| **Hyperparameter Tuning** | **EXCLUDED** | Hyperparameters (`n_estimators=120`, `max_depth=12`, `class_weight="balanced"`) were selected on Clip 4 cross-validation. |
| **Feature Selection & Scaler** | **EXCLUDED** | The 16 feature columns were defined from COCO keypoints and OpenSeeFace facial landmarks on Clip 4. |
| **Threshold Calibration** | **EXCLUDED** | Attention drop threshold (20.0%) and score calibration formula were tuned on Clip 4. |
| **Model Selection** | **EXCLUDED** | Model checkpoint selection occurred on Clip 4 without Clip 11 participation. |

**Verdict on Eligibility:** `data/clip_011.mp4` is **strictly held out and eligible** as an independent evaluation benchmark.

---

## 4. Ground-Truth Availability for Clip 11

We performed an exhaustive directory search across all XML, JSON, CSV, TXT, and YAML files in the workspace for ground truth labels corresponding to `clip_011.mp4`:
- Potential annotation files found: **0**
- `data/annotate for clip 4/annotations.xml` explicitly targets `clip 4.mp4` (21 tracks, 300 frames).
- No corresponding annotation task, XML file, or ground-truth label mapping exists for `clip_011.mp4`.

**Scientific Consequence:**
Because there are zero ground-truth labels for Clip 11:
- True Positives, False Positives, False Negatives, Accuracy, Precision, Recall, and Confusion Matrices cannot be computed mathematically.
- In accordance with research integrity rules, **classification benchmarking on Clip 11 is formally marked BLOCKED**.
- No synthetic ground-truth labels or group-level labels were fabricated to force an artificial benchmark.

---

## 5. Non-Destructive Live Execution Verification on Clip 11

While classification accuracy cannot be computed without ground truth, the non-LLM video pipeline was executed on `data/clip_011.mp4` to evaluate runtime software contracts:

- **Command Executed:**
  ```python
  pipeline = ClassroomAttentionPipeline(sample_fps=10.0, drop_threshold=20.0)
  results = pipeline.process_video(
      input_video_path=Path("data/clip_011.mp4"),
      output_video_path=Path("reports/audit/clip_011_audit_inference.mp4"),
      max_frames=30
  )
  ```
- **Execution Performance:** 30 frames processed in 21.3s (1.4 FPS on CUDA).
- **Student Census Count:** 22 seated students discovered and tracked.
- **Class Average Attention:** 67.0%.
- **Observation Breakdown:**
  - High Attention observations: 2 (0.3%)
  - Moderate Attention observations: 0 (0.0%)
  - Low Attention observations: 7 (1.1%)
  - Uncertain / Gated observations: 619 (98.6%)
  - Total observations: 628
- **Diagnostic Finding:** In `clip_011.mp4`, the elevated front-quarter camera angle causes a significant proportion of seated students to occupy $< 1\%$ of the frame area. The pipeline's distance gating module correctly assigned `is_distant=1.0` and categorized these observations as `Uncertain` rather than fabricating arbitrary gaze directions.

---

## 6. Detection, Tracking, and Classification Limitations

1. **Detection Attrition & Denominator Bias:**
   - In Clip 4, 357 ground-truth bounding boxes (5.7% of 6,300 boxes) were missed by YOLO11-Pose (or fell below IoU 0.20 threshold).
   - These 357 boxes were predominantly concentrated on specific students (Student #15 had only 25% detection coverage; Student #18 had 69%; Student #16 had 87%).
   - The reported 93.3% accuracy evaluated only the 5,943 *detected* boxes. If detection failures are penalized as classification failures, end-to-end system accuracy decreases to $\approx 88.0\%$.
2. **Track Fragmentation & ID Switches:**
   - Across Clip 4, 7 identity switches occurred.
   - When a student leans heavily across a desk or experiences temporary occlusion, the tracker enters `COASTING` mode. If coasting exceeds `max_disappeared_frames`, a new track ID is registered, fragmenting the student's historical timeline.
3. **Elevated Perspective Distance Gating:**
   - In Clip 11, back-row students possess facial regions smaller than $24 \times 24$ pixels. Landmark confidence falls below 0.40, correctly triggering distance gating (`is_distant=1.0`). While epistemically sound, it limits individual facial analysis to front-row and middle-row students.

---

## 7. Statistical Validity Audit (Before/After Assessment)

An empirical audit of the interval comparison module in `src/evaluation/before_after_analysis.py` identified critical statistical considerations:

### 7.1 The Pseudo-Replication Flaw in Frame-Level Pooling
When evaluating attention scores across two intervals (e.g. $[0\text{s}, 5\text{s}]$ vs $[5\text{s}, 10\text{s}]$), pooling all 10 Hz frame observations into a two-sample Welch's $t$-test introduces severe statistical invalidity:
1. **Violation of Independence ($i.i.d.$ Assumption):**
   - Observations sampled at 10 Hz from the same student exhibit strong temporal autocorrelation ($\text{AR}(1) \approx 0.85 - 0.95$).
   - A student observed across 50 frames produces 50 dependent data points, not 50 independent experiments.
2. **Artificial Degree-of-Freedom Inflation:**
   - Pooling 20 students over 100 frames generates $N \approx 2,000$ observation rows.
   - In our simulation audit:
     - **Frame-level pooled Welch's $t$-test:** $t = -5.48$, $p = 0.000000$ ($p < 10^{-6}$), falsely suggesting overwhelming significance.
     - **Student-level aggregated paired $t$-test:** $t = -2.68$, $p = 0.0147$ ($df = 19$).
   - While both detect the negative shift in this simulation, frame-level pooling inflates test statistics by a factor of $\sqrt{T/1} \approx \sqrt{50} \approx 7.07$, guaranteeing artificial statistical significance even on random noise drifts.

### 7.2 Methodologically Valid Statistical Alternatives
For academic defense and peer-reviewed publication, interval comparisons must utilize:
1. **Student-Level Mean Aggregation (Paired $t$-test / Wilcoxon Signed-Rank):**
   Collapse each student's temporal frames into a single pre-mean $\bar{X}_{i, \text{pre}}$ and post-mean $\bar{X}_{i, \text{post}}$. Test across the $N$ independent student units ($df = N - 1$).
2. **Linear Mixed-Effects Models (LMM):**
   $$\text{Attention}_{ijt} = \beta_0 + \beta_1 \cdot \text{Intervention}_{t} + u_i + \epsilon_{ijt}, \quad u_i \sim \mathcal{N}(0, \sigma_u^2)$$
   where $u_i$ is a random student intercept accounting for baseline individual differences, and $\epsilon_{ijt}$ follows an autoregressive $\text{AR}(1)$ error structure.

### 7.3 Epistemic Learning Gain Boundary
Attention score changes reflect **observable physical engagement markers** (head orientation, gaze deviations, posture slump, and motor activity). They **do not constitute direct proof of cognitive learning gains or academic mastery** without external educational assessments (e.g. pre/post quiz scores).

---

## 8. Exact Execution Commands & Results

### Checkpoint & Provenance Audit Command:
```powershell
python reports/audit/execute_independent_audit.py
```
**Execution Output:**
```text
================================================================================
EXECUTING INDEPENDENT MODEL QUALITY & BENCHMARK AUDIT
================================================================================

[1/5] Inspecting Checkpoint: final_attention_model.joblib...
  Model Architecture: RandomForestClassifier
  Classes: ['High', 'Low', 'Moderate']
  Feature Columns: 16 (Canonical 16: True)
  Trained Samples Count: 5732

[2/5] Auditing Training Provenance and Clip 11 Exclusion...
  Clip 11 Excluded from Model Fitting: True
  Clip 11 Individual Ground-Truth Labels Found: False
  Classification Benchmarking Status for Clip 11: BLOCKED (Missing individual ground truth labels for Clip 11)

[3/5] Auditing Existing Benchmark Artifacts (reports/independent_benchmark_metrics.json)...
  Video Evaluated in Existing Report: clip 4.mp4
  Reported 3-Class Accuracy: 93.3%
  True Scientific Characterization: Train-set re-substitution on Clip 4 (Not independent out-of-sample generalization)

[4/5] Auditing Statistical Validity of Before/After Assessment...
  Frame-level Pooled p-value: 0.000000 (Artificially deflated via pseudo-replication)
  Student-level Aggregated p-value: 0.014707 (Statistically sound df = 19)

[5/5] Running Non-Destructive Ingestion & Tracking Diagnostics on Clip 11...
  Processed 30 frames in 21.3s (1.4 FPS)
  Census Count: 22 seated students
  Class Average Attention: 67.0%
  Observation Breakdown: {'high_count': 2, 'moderate_count': 0, 'low_count': 7, 'uncertain_count': 619, 'total_observations': 628}

================================================================================
AUDIT COMPLETE. Summary saved to: reports/audit/independent_audit_summary.json
FINAL AUDIT VERDICT: BLOCKED
================================================================================
```

---

## 9. Final Verdict & Required Scientific Action

### Final Audit Status: **BLOCKED**

**Rationale:**
1. `data/clip_011.mp4` is legitimately held out from model training, hyperparameter tuning, and threshold calibration.
2. However, `data/clip_011.mp4` has **zero ground-truth attention annotations**.
3. Therefore, calculating true positive, false positive, accuracy, or F1 metrics on Clip 11 cannot be performed without fabricating ground-truth labels.
4. The reported 93.3% benchmark in `reports/independent_benchmark_metrics.json` was mathematically verified to be train-set re-substitution on `data/clip 4.mp4`.

---

## Single Most Important Next Action Required to Establish Scientific Validity

> [!IMPORTANT]
> **Primary Recommendation:**
> Create independent, peer-reviewed, multi-annotator CVAT bounding box and attention annotations (`High`, `Moderate`, `Low`) for **`data/clip_011.mp4`** (300 frames, 18–22 visible students). 
> 
> Once these labels are established and verified with inter-annotator agreement (Cohen's / Fleiss' kappa $\kappa \ge 0.75$), execute the frozen `final_attention_model.joblib` pipeline on Clip 11 to produce the **first scientifically defensible, out-of-sample generalization benchmark** for this thesis.
