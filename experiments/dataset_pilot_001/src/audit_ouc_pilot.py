import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, 
    recall_score, confusion_matrix
)

def run_experiment_c():
    start_time = time.time()
    out_dir = Path("experiments/dataset_pilot_001")
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("data/manifests/pilot_manifest.json")
    features_path = Path("data/processed/pilot_features.csv")

    print("[Exp C] Auditing Local OUC-CGE Pilot Data...")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    df_feats = pd.read_csv(features_path)
    print(f"  Loaded pilot_features.csv: {df_feats.shape[0]} student-track records across {df_feats['clip_id'].nunique()} clips")

    # Exclude confirmed frozen clips (view7, view8 in low) and duplicate (view2 in low)
    excluded_clips = {"view7", "view8"}
    
    # Filter dynamic clips
    dynamic_clips = []
    for c in manifest["clips"]:
        c_label = "high" if "high" in c["relative_path"] else "low"
        c_id = c["filename"].replace(".mp4", "")
        if c_label == "low" and c_id in excluded_clips:
            continue
        if c_label == "low" and c_id == "view2":
            # Duplicate of view1
            continue
        dynamic_clips.append({
            "session_id": c_label,
            "clip_id": c_id,
            "filename": c["filename"],
            "label": 1 if c_label == "high" else 0,
            "file_size_bytes": c["file_size_bytes"],
            "sha256": c["sha256"]
        })

    df_clips = pd.DataFrame(dynamic_clips)
    print(f"  Valid dynamic clips count: {len(df_clips)} (High: {(df_clips['label']==1).sum()}, Low: {(df_clips['label']==0).sum()})")

    # Aggregate student-track features to clip-level
    feature_cols = [
        "head_pitch_deg", "head_yaw_deg", "head_roll_deg", "head_pose_confidence",
        "gaze_horizontal", "gaze_vertical", "torso_angle_deg", "shoulder_tilt_deg",
        "movement_energy", "bbox_area_ratio", "detection_confidence"
    ]

    clip_agg = df_feats.groupby(["session_id", "clip_id"])[feature_cols].agg(["mean", "std"]).reset_index()
    clip_agg.columns = ["_".join(col).strip("_") for col in clip_agg.columns.values]
    
    # Merge with dynamic clips
    df_merged = pd.merge(df_clips, clip_agg, on=["session_id", "clip_id"], how="inner")
    print(f"  Feature-aggregated dynamic clips: {df_merged.shape[0]}")

    X_agg = df_merged[[c for c in df_merged.columns if "_mean" in c or "_std" in c]].fillna(0)
    y_agg = df_merged["label"]

    # 1. Leave-One-Out Cross-Validation for Clip-Level Classification
    loo = LeaveOneOut()
    pipe_lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    pipe_rf = Pipeline([("clf", RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42))])

    y_pred_lr = cross_val_predict(pipe_lr, X_agg, y_agg, cv=loo)
    y_pred_rf = cross_val_predict(pipe_rf, X_agg, y_agg, cv=loo)

    acc_lr = accuracy_score(y_agg, y_pred_lr)
    bal_lr = balanced_accuracy_score(y_agg, y_pred_lr)
    f1_lr = f1_score(y_agg, y_pred_lr, average="macro")
    cm_lr = confusion_matrix(y_agg, y_pred_lr).tolist()

    acc_rf = accuracy_score(y_agg, y_pred_rf)
    bal_rf = balanced_accuracy_score(y_agg, y_pred_rf)
    f1_rf = f1_score(y_agg, y_pred_rf, average="macro")
    cm_rf = confusion_matrix(y_agg, y_pred_rf).tolist()

    print(f"  Clip-Level Group Engagement Classification (N={len(df_merged)}):")
    print(f"    Logistic Regression LOO -> Acc: {acc_lr:.4f} | BalAcc: {bal_lr:.4f} | MacroF1: {f1_lr:.4f} | CM: {cm_lr}")
    print(f"    Random Forest LOO       -> Acc: {acc_rf:.4f} | BalAcc: {bal_rf:.4f} | MacroF1: {f1_rf:.4f} | CM: {cm_rf}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_lr, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax,
        xticklabels=["Low Engagement (0)", "High Engagement (1)"],
        yticklabels=["Low Engagement (0)", "High Engagement (1)"]
    )
    ax.set_title("Clip-Level Engagement Confusion Matrix (LOO-CV, N=21)\n[GROUP-LEVEL ATMOSPHERE ONLY]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Engagement Label", fontsize=10)
    ax.set_ylabel("True Group Engagement Label", fontsize=10)
    plt.tight_layout()
    cm_path = viz_dir / "confusion_matrix_ouc_group.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 2. Forensic Session Leakage Analysis
    leakage_analysis = {
        "finding": "SEVERE MULTI-VIEW SYNCHRONIZATION LEAKAGE",
        "description": "The 21 dynamic clips stem from only TWO physical classroom recordings: Session 1 (High engagement, views 1-12) and Session 2 (Low engagement, views 1-12). All high-engagement clips share the exact same room, lighting, instructor, and student cohort at the same time. Cross-validation across views inside the same session evaluates multi-camera angle consistency, NOT unseen-classroom generalization.",
        "individual_ground_truth_status": "ABSENT (Ecological Fallacy Risk)",
        "individual_ground_truth_note": "OUC-CGE provides ZERO individual student ground-truth attention annotations. Mapping group engagement (High/Low) to individual student tracks (1 to 6,298) is scientifically invalid."
    }

    metrics_ouc = {
        "experiment": "Experiment C: Existing OUC-CGE Pilot Audit",
        "dataset": "OUC-CGE (Ocean University of China Classroom Group Engagement)",
        "dataset_doi": "10.17605/OSF.IO/BRD2C",
        "license": "CC BY 4.0",
        "total_clips_present": len(manifest["clips"]),
        "valid_dynamic_clips_evaluated": len(df_merged),
        "excluded_frozen_clips": ["low/view7.mp4", "low/view8.mp4"],
        "excluded_duplicate_clips": ["low/view2.mp4 (byte identical to low/view1.mp4)"],
        "task": "Group-Level Engagement Classification (Clip Aggregation)",
        "models": {
            "LogisticRegression_LOO": {
                "accuracy": round(float(acc_lr), 4),
                "balanced_accuracy": round(float(bal_lr), 4),
                "macro_f1": round(float(f1_lr), 4),
                "confusion_matrix": cm_lr
            },
            "RandomForest_LOO": {
                "accuracy": round(float(acc_rf), 4),
                "balanced_accuracy": round(float(bal_rf), 4),
                "macro_f1": round(float(f1_rf), 4),
                "confusion_matrix": cm_rf
            }
        },
        "session_leakage_audit": leakage_analysis,
        "duration_sec": round(time.time() - start_time, 2)
    }

    metrics_file = out_dir / "metrics_ouc_audit.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_ouc, f, indent=2)
    print(f"  Saved Experiment C metrics to {metrics_file}")

if __name__ == "__main__":
    run_experiment_c()
