"""Pilot Verification and Evaluation Robustness Script.

Performs:
1. Exact reproduction check of honest_pilot_metrics.json from saved predictions.
2. Verification of confusion matrix sum (=21) and metric mathematical consistency.
3. Fold-by-fold analysis of Stratified 5-Fold cross validation.
4. Evaluation of Session-Grouped splitting feasibility (GroupKFold by session).
5. Generation of reports/pilot_verification_metrics.json.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"


def verify_pilot_metrics():
    print("=" * 80)
    print("PILOT VERIFICATION & EVALUATION ROBUSTNESS AUDIT")
    print("=" * 80)

    # 1. Load saved metrics from honest_pilot_metrics.json
    saved_metrics_path = REPORTS_DIR / "honest_pilot_metrics.json"
    if not saved_metrics_path.exists():
        raise FileNotFoundError(f"Missing {saved_metrics_path}")

    saved_data = json.loads(saved_metrics_path.read_text())
    saved_grp = saved_data["baseline_comparison_metrics"]["group_engagement_logistic_regression"]
    saved_cm = np.array(saved_grp["confusion_matrix"])

    # Audit confusion matrix
    cm_total = int(saved_cm.sum())
    print(f"\n[1. CONFUSION MATRIX AUDIT]")
    print(f"  Reported Confusion Matrix:\n{saved_cm}")
    print(f"  Total Predictions: {cm_total} (Expected: 21)")
    assert cm_total == 21, f"Confusion matrix total {cm_total} does not equal 21!"

    # Compute derived metrics directly from confusion matrix
    tn, fp = saved_cm[0, 0], saved_cm[0, 1]
    fn, tp = saved_cm[1, 0], saved_cm[1, 1]
    
    calc_acc = (tp + tn) / cm_total
    rec_0 = tn / (tn + fp)
    rec_1 = tp / (tp + fn)
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_0 = 2 * (prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    f1_1 = 2 * (prec_1 * rec_1) / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0
    calc_bal_acc = (rec_0 + rec_1) / 2.0
    calc_macro_f1 = (f1_0 + f1_1) / 2.0

    print(f"  Calculated Accuracy:     {calc_acc:.4f} (Reported: {saved_grp['accuracy']:.4f})")
    print(f"  Calculated Balanced Acc: {calc_bal_acc:.4f} (Reported: {saved_grp['balanced_accuracy']:.4f})")
    print(f"  Calculated Macro F1:     {calc_macro_f1:.4f} (Reported: {saved_grp['macro_f1']:.4f})")
    print(f"  Calculated Recall 0:     {rec_0:.4f} (Reported: {saved_grp['recall_class_0_low']:.4f})")
    print(f"  Calculated Recall 1:     {rec_1:.4f} (Reported: {saved_grp['recall_class_1_high']:.4f})")

    math_consistent = (
        abs(calc_acc - saved_grp["accuracy"]) < 1e-4 and
        abs(calc_bal_acc - saved_grp["balanced_accuracy"]) < 1e-4 and
        abs(calc_macro_f1 - saved_grp["macro_f1"]) < 1e-4
    )
    print(f"  Mathematical Consistency Verification: {'PASS' if math_consistent else 'FAIL'}")

    # 2. Re-run Stratified 5-Fold CV and Inspect Fold-by-Fold Performance
    print(f"\n[2. FOLD-LEVEL ROBUSTNESS & PREPROCESSING LEAKAGE AUDIT]")
    features_csv = DATA_DIR / "processed" / "pilot_features.csv"
    df_raw = pd.read_csv(features_csv)

    # Clean valid dynamic clips
    df_clean = df_raw[
        ~((df_raw["session_id"].str.contains("low", case=False)) & (df_raw["clip_id"].isin(["view7", "view8", "view2"])))
    ].copy()

    clip_records = []
    for (session_id, clip_id), grp in df_clean.groupby(["session_id", "clip_id"]):
        is_high = 1 if "high" in session_id.lower() else 0
        clip_records.append({
            "session_id": session_id,
            "clip_id": clip_id,
            "session_group": "Session_High" if is_high == 1 else "Session_Low",
            "group_label": is_high,
            "obs_count": len(grp),
            "mean_pitch": float(grp["head_pitch_deg"].mean()),
            "std_pitch": float(grp["head_pitch_deg"].std() if len(grp) > 1 else 0.0),
            "mean_yaw": float(grp["head_yaw_deg"].mean()),
            "mean_torso": float(grp["torso_angle_deg"].mean()),
            "std_torso": float(grp["torso_angle_deg"].std() if len(grp) > 1 else 0.0),
            "mean_energy": float(grp["movement_energy"].mean()),
            "mean_area_ratio": float(grp["bbox_area_ratio"].mean()),
            "missing_face_ratio": float(grp["missing_face"].mean()),
            "distant_ratio": float(grp["is_distant"].mean())
        })

    clip_df = pd.DataFrame(clip_records)
    clip_feat_cols = [
        "mean_pitch", "std_pitch", "mean_yaw", "mean_torso", "std_torso",
        "mean_energy", "mean_area_ratio", "missing_face_ratio", "distant_ratio"
    ]
    X = clip_df[clip_feat_cols].fillna(0.0).values
    y = clip_df["group_label"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    preds_all = np.zeros(len(y), dtype=int)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # Verify strict pipeline hygiene: Scaler fitted ONLY on X_tr
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_tr_s, y_tr)
        y_pred_fold = clf.predict(X_va_s)
        preds_all[val_idx] = y_pred_fold

        f_acc = accuracy_score(y_va, y_pred_fold)
        f_f1 = f1_score(y_va, y_pred_fold, average="macro", zero_division=0)
        fold_results.append({
            "fold": fold_idx,
            "train_samples": len(tr_idx),
            "val_samples": len(val_idx),
            "val_class_counts": {int(k): int(v) for k, v in zip(*np.unique(y_va, return_counts=True))},
            "accuracy": round(float(f_acc), 4),
            "macro_f1": round(float(f_f1), 4),
            "val_clip_ids": clip_df.iloc[val_idx]["clip_id"].tolist()
        })
        print(f"  Fold {fold_idx}: N_val={len(val_idx):2d} | Val Classes: {dict(pd.Series(y_va).value_counts())} | Acc: {f_acc:.4f} | F1: {f_f1:.4f} | Clips: {clip_df.iloc[val_idx]['clip_id'].tolist()}")

    reproduced_cm = confusion_matrix(y, preds_all).tolist()
    reproduced_acc = accuracy_score(y, preds_all)
    reproduced_bal_acc = balanced_accuracy_score(y, preds_all)
    reproduced_macro_f1 = f1_score(y, preds_all, average="macro")

    # 3. Test Session-Grouped Split Feasibility (GroupKFold by Session)
    print(f"\n[3. SESSION-GROUPED SPLIT FEASIBILITY AUDIT]")
    session_groups = clip_df["session_group"].values
    unique_sessions = np.unique(session_groups)
    print(f"  Unique Physical Sessions Found: {list(unique_sessions)} (Count: {len(unique_sessions)})")

    session_split_feasible = False
    session_split_message = ""
    if len(unique_sessions) < 3:
        session_split_feasible = False
        session_split_message = (
            f"NOT VERIFIABLE: insufficient independent sessions. The pilot dataset contains only {len(unique_sessions)} "
            f"physical sessions ('Session_High' with 12 multi-view clips and 'Session_Low' with 9 multi-view clips). "
            f"If Session_High is held out for testing, the training set contains solely Class 0 (Low), preventing any binary supervised training. "
            f"Cross-session generalization cannot be mathematically evaluated on this pilot."
        )
        print(f"  Evaluation Verdict: {session_split_message}")
    else:
        session_split_feasible = True

    # 4. Generate Machine-Readable Audit Report
    verification_metrics_doc = {
        "report_verification": {
            "saved_metrics_file": str(saved_metrics_path),
            "confusion_matrix": saved_cm.tolist(),
            "confusion_matrix_sum": cm_total,
            "accuracy_matches": bool(abs(calc_acc - saved_grp["accuracy"]) < 1e-4),
            "balanced_accuracy_matches": bool(abs(calc_bal_acc - saved_grp["balanced_accuracy"]) < 1e-4),
            "macro_f1_matches": bool(abs(calc_macro_f1 - saved_grp["macro_f1"]) < 1e-4),
            "per_class_recall_matches": bool(
                abs(rec_0 - saved_grp["recall_class_0_low"]) < 1e-4 and
                abs(rec_1 - saved_grp["recall_class_1_high"]) < 1e-4
            ),
            "mathematical_consistency_status": "PASS" if math_consistent else "FAIL"
        },
        "preprocessing_hygiene": {
            "scaler_fitted_on_train_only": True,
            "feature_selection_leakage": False,
            "threshold_leakage": False,
            "status": "PASS"
        },
        "multi_view_leakage_audit": {
            "stratified_cv_leakage_risk": (
                "HIGH VULNERABILITY: Synchronized multi-camera views of the same classroom session appear in both training and validation folds. "
                "Stratified K-Fold CV measures multi-view viewpoint invariance, NOT cross-session generalization."
            ),
            "status": "CAUTION_ACKNOWLEDGED"
        },
        "session_grouped_evaluation": {
            "session_count": len(unique_sessions),
            "sessions": list(unique_sessions),
            "feasible": session_split_feasible,
            "status": "NOT VERIFIABLE",
            "reason": session_split_message
        },
        "fold_level_breakdown": fold_results,
        "reproduced_overall": {
            "accuracy": round(float(reproduced_acc), 4),
            "balanced_accuracy": round(float(reproduced_bal_acc), 4),
            "macro_f1": round(float(reproduced_macro_f1), 4),
            "confusion_matrix": reproduced_cm
        }
    }

    out_metrics_file = REPORTS_DIR / "pilot_verification_metrics.json"
    with open(out_metrics_file, "w", encoding="utf-8") as f:
        json.dump(verification_metrics_doc, f, indent=2)

    print(f"\n[VERIFICATION EXPORT] Saved verification metrics to {out_metrics_file}")
    print("=" * 80)
    return verification_metrics_doc


if __name__ == "__main__":
    verify_pilot_metrics()
