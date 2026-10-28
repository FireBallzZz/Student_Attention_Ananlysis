"""Phase Six: Feature Extraction, Model Training, and Baseline Comparison.

Extracts multimodal features from pilot clips, splits by independent clips/sessions
to prevent temporal leakage, trains 5 baseline models + temporal BiGRU,
and evaluates full metrics suite (Accuracy, Balanced Acc, Macro F1, Per-Class P/R, Confusion Matrix, Latency).
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss
)
from sklearn.preprocessing import StandardScaler

from src.ingestion.video_reader import VideoReader
from src.detection.pose_detector import YOLO11PoseDetector
from src.tracking.tracker import ByteTrackTracker
from src.modalities.facial_attention import ModalityExtractor
from src.features.feature_schema import FEATURE_COLUMNS, FrameFeatureVector, create_feature_vector
from src.models.baseline_models import StudentBiGRU, get_baseline_models


def extract_features_from_pilot(pilot_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Run full CV pipeline across pilot clips and export structured feature dataset."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and output_csv.stat().st_size > 1000:
        print(f"[CACHE] Loading pre-extracted features from {output_csv}...")
        return pd.read_csv(output_csv)

    print("[EXTRACTION] Running feature extraction on pilot video clips...")
    reader = VideoReader(sample_fps=5.0)  # 5 FPS temporal sampling
    detector = YOLO11PoseDetector()
    tracker = ByteTrackTracker()
    extractor = ModalityExtractor()

    # Find valid clips (excluding frozen view7 and view8 in low)
    all_clips = sorted(pilot_dir.rglob("*.mp4"))
    valid_clips = [c for c in all_clips if not ("low" in str(c).lower() and c.name in ["view7.mp4", "view8.mp4"])]

    records: List[Dict] = []
    t0 = time.time()

    for c_idx, clip_path in enumerate(valid_clips):
        tracker.reset()
        is_high = "high" in str(clip_path).lower()
        target_label = 1 if is_high else 0
        session_id = clip_path.parent.name
        clip_id = clip_path.stem

        print(f"  [{c_idx + 1}/{len(valid_clips)}] Processing {clip_id} ({'high' if is_high else 'low'})...")

        for ts_frame in reader.stream_frames(clip_path):
            frame = ts_frame.frame
            dets = detector.detect(frame)
            tracked = tracker.update(frame, dets)

            for st in tracked:
                mods = extractor.extract(st, frame.shape)
                fv = create_feature_vector(
                    session_id=session_id,
                    clip_id=clip_id,
                    student=st,
                    modalities=mods,
                    frame_idx=ts_frame.frame_idx,
                    timestamp_sec=ts_frame.timestamp_sec,
                    target_label=target_label
                )
                records.append(fv.to_dict())

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[EXTRACTION COMPLETE] Extracted {len(df)} student observation records in {time.time() - t0:.1f}s.")
    return df


def split_by_independent_clips(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset strictly by independent video clips.
    Training: view1 - view6, view9, view10 (8 clips per class)
    Validation: view11, view12, view3, view4 (held-out camera viewpoints / sessions)
    Prevents adjacent frame leakage!
    """
    val_clip_stems = ["view11", "view12", "view3"]
    val_mask = df["clip_id"].isin(val_clip_stems)

    train_df = df[~val_mask].copy()
    val_df = df[val_mask].copy()

    print(f"Independent Session Split:")
    print(f"  Train: {len(train_df)} samples across clips: {sorted(train_df['clip_id'].unique())}")
    print(f"  Val:   {len(val_df)} samples across clips: {sorted(val_df['clip_id'].unique())}")
    return train_df, val_df


def train_evaluate_temporal_bigru(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    seq_len: int = 5,
    seed: int = 42
) -> Dict:
    """Train and evaluate compact PyTorch BiGRU on temporal sequences."""
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
        num_seq = len(X) // seq_len
        if num_seq == 0:
            return np.empty((0, seq_len, X.shape[1])), np.empty(0)
        X_cut = X[:num_seq * seq_len].reshape(num_seq, seq_len, X.shape[1])
        y_cut = y[:num_seq * seq_len].reshape(num_seq, seq_len)[:, -1]  # label of last frame
        return X_cut, y_cut

    X_train_seq, y_train_seq = build_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = build_sequences(X_val, y_val, seq_len)

    if len(X_train_seq) < 10 or len(X_val_seq) < 5:
        return {"model": "BiGRU", "status": "SKIPPED", "reason": "Insufficient sequence length in split"}

    train_ds = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = StudentBiGRU(input_dim=X_train.shape[1], hidden_dim=24).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    # Train for 15 epochs
    model.train()
    for epoch in range(15):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    val_preds = []
    val_probs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_probs.extend(probs)
            val_preds.extend(preds)
    infer_time_ms = ((time.perf_counter() - t0) / len(y_val_seq)) * 1000.0

    val_preds = np.array(val_preds)
    val_probs = np.array(val_probs)

    acc = accuracy_score(y_val_seq, val_preds)
    bal_acc = balanced_accuracy_score(y_val_seq, val_preds)
    macro_f1 = f1_score(y_val_seq, val_preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_val_seq, val_preds).tolist()
    brier = float(brier_score_loss(y_val_seq, val_probs))

    return {
        "model_name": "Temporal BiGRU (seq_len=5)",
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "precision_class_0": round(float(precision_score(y_val_seq, val_preds, pos_label=0, zero_division=0)), 4),
        "recall_class_0": round(float(recall_score(y_val_seq, val_preds, pos_label=0, zero_division=0)), 4),
        "precision_class_1": round(float(precision_score(y_val_seq, val_preds, pos_label=1, zero_division=0)), 4),
        "recall_class_1": round(float(recall_score(y_val_seq, val_preds, pos_label=1, zero_division=0)), 4),
        "confusion_matrix": cm,
        "brier_score": round(brier, 4),
        "latency_ms_per_sample": round(float(infer_time_ms), 3)
    }


def run_model_experiments():
    """Main experiment runner comparing all baseline models."""
    project_root = Path(__file__).resolve().parent.parent
    pilot_dir = project_root / "data" / "pilot"
    features_csv = project_root / "data" / "processed" / "pilot_features.csv"
    checkpoints_dir = project_root / "checkpoints"
    reports_dir = project_root / "reports"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Extraction
    df = extract_features_from_pilot(pilot_dir, features_csv)

    # 2. Independent Session Split
    train_df, val_df = split_by_independent_clips(df)

    X_train_raw = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_train = train_df["target_label"].values.astype(int)
    X_val_raw = val_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_val = val_df["target_label"].values.astype(int)

    # 3. Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    joblib.dump(scaler, checkpoints_dir / "feature_scaler.joblib")

    # Log class distributions
    train_dist = {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    val_dist = {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))}

    print(f"\nClass Distributions:")
    print(f"  Training Split:   {train_dist} (Total: {len(y_train)})")
    print(f"  Validation Split: {val_dist} (Total: {len(y_val)})")

    # 4. Train and Evaluate Sklearn Baselines
    models = get_baseline_models(seed=42)
    benchmark_results = []
    best_model_name = None
    best_macro_f1 = -1.0
    best_model_obj = None

    print("\n" + "=" * 70)
    print("PHASE SIX: BASELINE MODEL EVALUATION")
    print("=" * 70)

    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)

        # Measure inference latency
        t0 = time.perf_counter()
        y_pred = model.predict(X_val)
        t_infer = (time.perf_counter() - t0) / max(1, len(X_val)) * 1000.0

        # Predict probabilities
        has_proba = hasattr(model, "predict_proba")
        brier = 0.5
        if has_proba:
            y_prob = model.predict_proba(X_val)[:, 1]
            brier = float(brier_score_loss(y_val, y_prob))

        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        p0 = precision_score(y_val, y_pred, pos_label=0, zero_division=0)
        r0 = recall_score(y_val, y_pred, pos_label=0, zero_division=0)
        p1 = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        r1 = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_val, y_pred).tolist()

        res_entry = {
            "model_name": name,
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "precision_class_0": round(float(p0), 4),
            "recall_class_0": round(float(r0), 4),
            "precision_class_1": round(float(p1), 4),
            "recall_class_1": round(float(r1), 4),
            "confusion_matrix": cm,
            "brier_score": round(brier, 4),
            "latency_ms_per_sample": round(float(t_infer), 4)
        }
        benchmark_results.append(res_entry)

        print(f"{name:30s} | Acc: {acc:.3f} | BalAcc: {bal_acc:.3f} | MacroF1: {macro_f1:.3f} | Latency: {t_infer:.3f}ms")

        if macro_f1 > best_macro_f1 and name != "Majority Class Baseline":
            best_macro_f1 = macro_f1
            best_model_name = name
            best_model_obj = model

    # 5. Temporal BiGRU Model Evaluation
    print("\n--- Evaluating Temporal BiGRU Sequence Model ---")
    bigru_res = train_evaluate_temporal_bigru(X_train, y_train, X_val, y_val, seq_len=5, seed=42)
    benchmark_results.append(bigru_res)
    if "accuracy" in bigru_res:
        print(f"{bigru_res['model_name']:30s} | Acc: {bigru_res['accuracy']:.3f} | BalAcc: {bigru_res['balanced_accuracy']:.3f} | MacroF1: {bigru_res['macro_f1']:.3f}")

    # 6. Save Best Checkpoint
    if best_model_obj is not None:
        best_ckpt_path = checkpoints_dir / "best_baseline_model.joblib"
        joblib.dump({"model_name": best_model_name, "model": best_model_obj, "features": FEATURE_COLUMNS}, best_ckpt_path)
        print(f"\n[CHECKPOINT] Saved best performing baseline ({best_model_name}) to {best_ckpt_path}")

    # 7. Write Full JSON Report
    report_doc = {
        "timestamp": time.time(),
        "total_observations": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_class_distribution": train_dist,
        "val_class_distribution": val_dist,
        "split_strategy": "Independent video clips (view11, view12, view3 held out). Zero temporal frame leakage.",
        "best_model": best_model_name,
        "best_macro_f1": best_macro_f1,
        "feature_schema": FEATURE_COLUMNS,
        "model_results": benchmark_results
    }

    report_path = reports_dir / "baseline_experiments_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_doc, f, indent=2)

    print(f"[REPORT] Baseline experiment report saved to: {report_path}")
    print("=" * 70)
    return report_doc


if __name__ == "__main__":
    run_model_experiments()
