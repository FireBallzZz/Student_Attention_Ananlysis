# Exact Run Commands for Dataset Pilot Reproducibility

**Run Directory**: `experiments/dataset_pilot_001/`  
**Execution Shell**: PowerShell (Windows 11)  
**Python Version**: 3.14.4  

---

## 1. System & Environment Verification

```powershell
# Navigate to workspace root
cd "c:\Users\User\Downloads\Model v2\classroom_attention_research"

# Inspect Python, GPU, CUDA, and library versions
python scratch/inspect_env.py
```

---

## 2. Experiment A: Tabular Attention Classification (Mendeley `smzggbnkd2/3`)

```powershell
# Step 2.1: Download official Mendeley v3 feature dataset (1.53 MB, CC BY 4.0)
python scratch/download_mendeley.py

# Step 2.2: Audit Mendeley data schema, missing values, duplicates, and target distribution
python scratch/audit_mendeley_data.py

# Step 2.3: Train baseline classifiers (Dummy, Logistic Regression, Random Forest, HistGradientBoosting)
# Evaluates 5-fold CV on train partition (N=1,050) and held-out test evaluation (N=450)
python experiments/dataset_pilot_001/src/train_tabular_pilot.py
```

**Artifacts Generated**:
- Model Checkpoint: `experiments/dataset_pilot_001/models/best_tabular_model.joblib`
- Feature Schema: `experiments/dataset_pilot_001/feature_schema.json`
- Split Manifest: `experiments/dataset_pilot_001/split_manifest.json`
- Metrics JSON: `experiments/dataset_pilot_001/metrics_tabular.json`
- Confusion Matrix: `experiments/dataset_pilot_001/visualizations/confusion_matrix_tabular.png`

---

## 3. Experiment B: Student Action Detection Pilot (SCB-Dataset)

```powershell
# Step 3.1: Download small SCB-Dataset pilot (100 images: 70 train, 30 val) via seekable HTTP range
python scratch/download_scb_pilot.py

# Step 3.2: Fine-tune YOLO11n for 5 epochs with conservative settings
python experiments/dataset_pilot_001/src/train_scb_detector.py
```

**Artifacts Generated**:
- Best Checkpoint: `experiments/dataset_pilot_001/models/best_scb_yolo11n.pt`
- Download Manifest: `experiments/dataset_pilot_001/data/scb_pilot/download_manifest.json`
- Metrics JSON: `experiments/dataset_pilot_001/metrics_scb_detector.json`
- Visual GT vs. Pred: `experiments/dataset_pilot_001/visualizations/scb_gt_vs_pred.png`

---

## 4. Experiment C: Existing OUC-CGE Pilot Audit & Session Leakage Analysis

```powershell
# Run independent audit of 21 dynamic clips, exclude frozen/duplicate clips, and evaluate group engagement
python experiments/dataset_pilot_001/src/audit_ouc_pilot.py
```

**Artifacts Generated**:
- Metrics JSON: `experiments/dataset_pilot_001/metrics_ouc_audit.json`
- Confusion Matrix: `experiments/dataset_pilot_001/visualizations/confusion_matrix_ouc_group.png`

---

## 5. Master Report & Metric Compilation

```powershell
# Compile unified metrics.json and final audit summaries
python experiments/dataset_pilot_001/src/compile_master_metrics.py
```
