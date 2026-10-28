import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, 
    recall_score, confusion_matrix, classification_report
)

def run_experiment_a():
    np.random.seed(42)
    start_time_total = time.time()
    
    out_dir = Path("experiments/dataset_pilot_001")
    models_dir = out_dir / "models"
    viz_dir = out_dir / "visualizations"
    models_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "data" / "mendeley" / "attention_detection_dataset_v3.csv"
    print(f"[Exp A] Loading Mendeley dataset from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    print(f"  Raw dataset shape: {df_raw.shape}")

    # Step 1: Stratified sampling of 1,500 rows for the pilot (target: 1,000–2,000 rows)
    TARGET_COL = "label"
    PILOT_SIZE = 1500
    
    df_pilot, _ = train_test_split(
        df_raw, train_size=PILOT_SIZE, stratify=df_raw[TARGET_COL], random_state=42
    )
    df_pilot = df_pilot.reset_index(drop=True)
    
    print(f"  Sampled pilot dataset shape: {df_pilot.shape}")
    print(f"  Pilot class distribution:\n{df_pilot[TARGET_COL].value_counts().to_dict()}")

    # Step 2: Separate features and target
    X = df_pilot.drop(columns=[TARGET_COL])
    y = df_pilot[TARGET_COL]

    # Identify column types
    categorical_cols = ["head_pose", "gaze_direction"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    print(f"  Numeric features ({len(numeric_cols)}): {numeric_cols[:5]}...")
    print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Step 3: Train / Test Split (70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"  Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Save split manifest
    split_manifest = {
        "dataset_name": "Mendeley Students Attention Detection Dataset (v3)",
        "doi": "10.17632/smzggbnkd2.3",
        "pilot_total_samples": len(df_pilot),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_seed": 42,
        "stratified": True,
        "train_class_counts": y_train.value_counts().to_dict(),
        "test_class_counts": y_test.value_counts().to_dict(),
        "train_indices": X_train.index.tolist(),
        "test_indices": X_test.index.tolist()
    }
    with open(out_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)

    # Step 4: Build Preprocessing Pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    # Save feature schema
    schema = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target": TARGET_COL,
        "classes": {0: "Attentive", 1: "Inattentive"}
    }
    with open(out_dir / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # Step 5: Define Models
    models = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=42)
    }

    results = {}
    best_macro_f1 = -1.0
    best_model_name = None
    best_pipeline = None

    # Step 6: 5-Fold Stratified Cross-Validation on Train Partition
    print("\n--- Running 5-Fold Stratified Cross-Validation on Train Split ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scoring = ["accuracy", "balanced_accuracy", "f1_macro"]

    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=cv_scoring, n_jobs=-1)
        print(f"  {name:22s} | CV Acc: {cv_res['test_accuracy'].mean():.4f} +/- {cv_res['test_accuracy'].std():.4f} | CV BalAcc: {cv_res['test_balanced_accuracy'].mean():.4f} | CV MacroF1: {cv_res['test_f1_macro'].mean():.4f}")

    # Step 7: Train on Full Train Split, Evaluate on Held-Out Test Split
    print("\n--- Evaluating on Held-Out Test Partition (N=450) ---")
    test_metrics = {}

    for name, clf in models.items():
        t0 = time.time()
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0

        t_inf0 = time.time()
        y_pred = pipe.predict(X_test)
        inf_time = time.time() - t_inf0

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        p0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        r0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

        p1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        r1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        cm = confusion_matrix(y_test, y_pred).tolist()

        test_metrics[name] = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "class_0_attentive": {
                "precision": round(float(p0), 4),
                "recall": round(float(r0), 4),
                "f1": round(float(f0), 4)
            },
            "class_1_inattentive": {
                "precision": round(float(p1), 4),
                "recall": round(float(r1), 4),
                "f1": round(float(f1), 4)
            },
            "confusion_matrix": cm,
            "train_duration_sec": round(float(fit_time), 4),
            "inference_duration_sec": round(float(inf_time), 4)
        }

        print(f"  {name:22s} | Test Acc: {acc:.4f} | BalAcc: {bal_acc:.4f} | MacroF1: {macro_f1:.4f} | Time: {fit_time:.3f}s")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = name
            best_pipeline = pipe
            best_y_pred = y_pred

    print(f"\nBest Model Selected: {best_model_name} (Macro F1 = {best_macro_f1:.4f})")

    # Step 8: Save Best Model Checkpoint
    model_path = models_dir / "best_tabular_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"  Saved best model checkpoint to {model_path}")

    # Step 9: Generate Confusion Matrix Visualization
    cm = confusion_matrix(y_test, best_y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Attentive (0)", "Inattentive (1)"],
        yticklabels=["Attentive (0)", "Inattentive (1)"]
    )
    ax.set_title(f"Confusion Matrix: {best_model_name}\n(Held-out Test Set, N=450)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Ground-Truth Label", fontsize=11)
    plt.tight_layout()
    cm_path = viz_dir / "confusion_matrix_tabular.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix plot to {cm_path}")

    # Step 10: Save Metrics
    total_elapsed = time.time() - start_time_total
    final_output = {
        "experiment": "Experiment A: Tabular Attention Classification",
        "dataset": "Mendeley Students Attention Detection Dataset v3",
        "dataset_doi": "10.17632/smzggbnkd2.3",
        "pilot_sample_size": PILOT_SIZE,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "best_model": best_model_name,
        "models_evaluated": test_metrics,
        "total_experiment_duration_sec": round(total_elapsed, 2)
    }

    metrics_file = out_dir / "metrics_tabular.json"
    with open(metrics_file, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"  Saved Experiment A metrics to {metrics_file}")

if __name__ == "__main__":
    run_experiment_a()
