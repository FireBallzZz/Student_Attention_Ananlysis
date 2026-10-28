import os
import sys
import time
import json
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO

def verify_dataset(data_dir):
    data_path = Path(data_dir)
    train_imgs = list((data_path / "images" / "train").glob("*.*"))
    val_imgs = list((data_path / "images" / "val").glob("*.*"))
    train_lbls = list((data_path / "labels" / "train").glob("*.txt"))
    val_lbls = list((data_path / "labels" / "val").glob("*.txt"))

    print(f"Dataset Verification:")
    print(f"  Train: {len(train_imgs)} images, {len(train_lbls)} labels")
    print(f"  Val:   {len(val_imgs)} images, {len(val_lbls)} labels")

    # Count classes in annotations
    class_counts = {0: 0, 1: 0, 2: 0} # 0: hand-raising, 1: read, 2: write
    class_names = {0: "hand-raising", 1: "read", 2: "write"}

    for lbl in train_lbls + val_lbls:
        lines = lbl.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.split()
            if parts:
                cid = int(parts[0])
                class_counts[cid] = class_counts.get(cid, 0) + 1

    print(f"  Total annotated object instances:")
    for cid, cnt in class_counts.items():
        print(f"    Class {cid} ({class_names.get(cid, 'unknown')}): {cnt} bounding boxes")

    return {
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
        "class_instance_counts": {class_names.get(k, str(k)): v for k, v in class_counts.items()}
    }

def visualize_sample_prediction(model, val_img_path, val_lbl_path, out_path, class_names):
    img = cv2.imread(str(val_img_path))
    if img is None:
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Parse ground truth
    gt_boxes = []
    if val_lbl_path.exists():
        lines = val_lbl_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cid = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                gt_boxes.append((cid, x1, y1, bw * w, bh * h))

    # Run model prediction
    results = model.predict(str(val_img_path), conf=0.25, imgsz=640, verbose=False)[0]
    pred_boxes = []
    for box in results.boxes:
        cid = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        pred_boxes.append((cid, conf, x1, y1, x2 - x1, y2 - y1))

    # Create figure with 2 subplots: GT vs Predicted
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    ax1.imshow(img_rgb)
    ax1.set_title(f"Ground Truth Annotations ({len(gt_boxes)} boxes)", fontsize=12, fontweight="bold")
    ax1.axis("off")
    for cid, x, y, bw, bh in gt_boxes:
        rect = patches.Rectangle((x, y), bw, bh, linewidth=2, edgecolor="lime", facecolor="none")
        ax1.add_patch(rect)
        cname = class_names.get(cid, str(cid))
        ax1.text(x, max(0, y - 5), cname, color="white", fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="square,pad=0.2", facecolor="green", alpha=0.8))

    ax2.imshow(img_rgb)
    ax2.set_title(f"YOLO11n Predicted Detections ({len(pred_boxes)} boxes)", fontsize=12, fontweight="bold")
    ax2.axis("off")
    for cid, conf, x, y, bw, bh in pred_boxes:
        rect = patches.Rectangle((x, y), bw, bh, linewidth=2, edgecolor="deepskyblue", facecolor="none")
        ax2.add_patch(rect)
        cname = class_names.get(cid, str(cid))
        ax2.text(x, max(0, y - 5), f"{cname} {conf:.2f}", color="white", fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="square,pad=0.2", facecolor="blue", alpha=0.8))

    plt.suptitle("SCB-Dataset Pilot: Ground Truth vs. Fine-Tuned YOLO11n Detections\n[PIPELINE SMOKE TEST ONLY]", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved visual comparison to {out_path}")

def run_experiment_b():
    start_time = time.time()
    out_dir = Path("experiments/dataset_pilot_001")
    models_dir = out_dir / "models"
    viz_dir = out_dir / "visualizations"
    models_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = out_dir / "data" / "scb_pilot" / "scb_pilot.yaml"
    print(f"[Exp B] Starting Classroom Action Detection Pilot on SCB-Dataset...")
    print(f"  Dataset YAML: {yaml_path}")

    # 1. Dataset Verification
    audit_data = verify_dataset(out_dir / "data" / "scb_pilot")

    # 2. Check GPU Memory before training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"  Initial GPU Memory Allocated: {mem_before_mb:.2f} MB")
    else:
        mem_before_mb = 0.0

    # 3. Initialize YOLO11n model
    print("  Initializing YOLO11n pretrained baseline...")
    model = YOLO("yolo11n.pt")

    # 4. Conservative fine-tuning settings (Smoke test: 5 epochs, batch 8, early stopping)
    runs_dir = out_dir / "yolo_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("  Beginning conservative fine-tuning (epochs=5, imgsz=640, batch=8)...")
    train_t0 = time.time()
    train_results = model.train(
        data=str(yaml_path),
        epochs=5,
        imgsz=640,
        batch=8,
        workers=2,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(runs_dir),
        name="scb_pilot_smoke_test",
        patience=3,
        save=True,
        verbose=True
    )
    train_duration = time.time() - train_t0
    print(f"  Training completed in {train_duration:.2f} seconds.")

    # 5. Peak GPU Memory
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"  Peak GPU Memory Used: {peak_gpu_mb:.2f} MB")
    else:
        peak_gpu_mb = 0.0

    # 6. Evaluation on Validation Set
    print("\n  Evaluating fine-tuned model on validation set...")
    val_t0 = time.time()
    val_results = model.val(data=str(yaml_path), imgsz=640, batch=8, verbose=True)
    val_duration = time.time() - val_t0

    # Extract metrics
    class_names = {0: "hand-raising", 1: "read", 2: "write"}
    mp = float(val_results.box.mp)
    mr = float(val_results.box.mr)
    map50 = float(val_results.box.map50)
    map = float(val_results.box.map)

    print(f"  Validation Results:")
    print(f"    mAP@0.5:      {map50:.4f}")
    print(f"    mAP@0.5:0.95: {map:.4f}")
    print(f"    Precision:    {mp:.4f}")
    print(f"    Recall:       {mr:.4f}")

    # Per-class metrics if available
    per_class_metrics = {}
    if hasattr(val_results.box, "p") and len(val_results.box.p) == 3:
        for cid in range(3):
            cname = class_names[cid]
            per_class_metrics[cname] = {
                "precision": round(float(val_results.box.p[cid]), 4),
                "recall": round(float(val_results.box.r[cid]), 4),
                "map50": round(float(val_results.box.ap50[cid]), 4),
                "map": round(float(val_results.box.ap[cid]), 4)
            }
            print(f"    Class '{cname}': P={val_results.box.p[cid]:.4f}, R={val_results.box.r[cid]:.4f}, mAP50={val_results.box.ap50[cid]:.4f}")

    # 7. Visual Comparison on Sample Val Image
    val_imgs = list((out_dir / "data" / "scb_pilot" / "images" / "val").glob("*.*"))
    if val_imgs:
        sample_img = val_imgs[0]
        sample_lbl = out_dir / "data" / "scb_pilot" / "labels" / "val" / f"{sample_img.stem}.txt"
        viz_out = viz_dir / "scb_gt_vs_pred.png"
        visualize_sample_prediction(model, sample_img, sample_lbl, viz_out, class_names)

    # 8. Save Best Checkpoint
    best_pt = runs_dir / "scb_pilot_smoke_test" / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        saved_ckpt = models_dir / "best_scb_yolo11n.pt"
        shutil.copy(best_pt, saved_ckpt)
        print(f"  Saved best SCB model checkpoint to {saved_ckpt}")

    # 9. Save Metrics JSON
    final_metrics = {
        "experiment": "Experiment B: Classroom Behavior Detection",
        "dataset": "SCB-Dataset (SCB5 Handrise-Read-Write Pilot)",
        "dataset_url": "https://github.com/Whiffe/SCB-dataset",
        "result_type": "Pipeline Smoke Test (Small Pilot)",
        "train_images": audit_data["train_images"],
        "val_images": audit_data["val_images"],
        "class_instance_counts": audit_data["class_instance_counts"],
        "model_architecture": "YOLO11n (Ultralytics 8.4.182)",
        "epochs_trained": 5,
        "overall_metrics": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "mAP_50": round(map50, 4),
            "mAP_50_95": round(map, 4)
        },
        "per_class_metrics": per_class_metrics,
        "timing_and_hardware": {
            "train_duration_sec": round(train_duration, 2),
            "val_duration_sec": round(val_duration, 2),
            "peak_gpu_memory_mb": round(peak_gpu_mb, 2),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "scientific_disclaimer": "Physical actions (hand-raising, reading, writing) do NOT constitute definitive attention/inattention ground truth. Hand-raising is an active participatory action; reading/writing are learning tasks. This pilot benchmarks action detection capabilities, not cognitive mental states."
    }

    metrics_path = out_dir / "metrics_scb_detector.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"  Saved Experiment B metrics to {metrics_path}")

if __name__ == "__main__":
    run_experiment_b()
