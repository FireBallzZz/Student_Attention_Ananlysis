"""Training script for Final Calibrated Classroom Attention Model.

Extracts multimodal features from verified ground-truth student annotations (Clip 4),
trains a balanced ensemble classifier, and exports the final model artifact.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time
import json
import joblib
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score

from src.detection.pose_detector import YOLO11PoseDetector
from src.tracking.tracker import ClassroomCensusTracker
from src.modalities.face_cropper import FaceCropper
from src.modalities.openseeface_tracker import OpenSeeFaceTracker
from src.modalities.facial_attention import ModalityExtractor
from src.features.feature_schema import FEATURE_COLUMNS, create_feature_vector
from scratch.evaluate_clip4_accuracy import compute_iou, load_ground_truth

def train_and_save():
    xml_path = Path("data/annotate for clip 4/annotations.xml")
    video_path = Path("data/clip 4.mp4")
    checkpoints_dir = Path("classroom_attention_research/checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = checkpoints_dir / "final_attention_model.joblib"
    
    gt_frames, gt_tracks = load_ground_truth(xml_path)
    cap = cv2.VideoCapture(str(video_path))
    
    detector = YOLO11PoseDetector(conf_threshold=0.15, imgsz=1280)
    tracker = ClassroomCensusTracker()
    face_cropper = FaceCropper()
    osf_tracker = OpenSeeFaceTracker()
    mod_extractor = ModalityExtractor()
    
    records = []
    frame_idx = 0
    t0 = time.time()
    
    print(f"\n[1/3] Extracting multimodal features from {video_path.name}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur_f = frame_idx
        timestamp_sec = cur_f / 29.97
        frame_idx += 1
        
        gts = gt_frames.get(cur_f, [])
        if not gts:
            continue
            
        dets = detector.detect(frame)
        students = tracker.update(frame, dets)
        
        for gt in gts:
            best_s = None
            best_iou = 0.0
            for s in students:
                iou = compute_iou(gt["bbox"], s.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_s = s
                    
            if best_s and best_iou >= 0.30:
                crop_res = face_cropper.crop(frame, best_s.bbox, best_s.keypoints_xy, best_s.keypoints_conf)
                osf_res = osf_tracker.process_face(best_s.track_id, crop_res, best_s.keypoints_xy, best_s.keypoints_conf)
                mods = mod_extractor.extract(best_s, frame.shape)
                
                fv = create_feature_vector(
                    session_id="clip_4",
                    clip_id="clip_4",
                    student=best_s,
                    modalities=mods,
                    frame_idx=cur_f,
                    timestamp_sec=timestamp_sec,
                    target_label=1
                )
                
                feat_dict = {col: getattr(fv, col) for col in FEATURE_COLUMNS}
                feat_dict["label"] = gt["attention"]
                feat_dict["gt_track_id"] = gt["track_id"]
                feat_dict["frame_idx"] = cur_f
                records.append(feat_dict)
                
    cap.release()
    df = pd.DataFrame(records)
    print(f"Extraction complete in {time.time() - t0:.1f}s: {len(df)} student-frame samples.")
    print("Class distribution:\n", df["label"].value_counts())
    
    X = np.array(df[FEATURE_COLUMNS].values, dtype=np.float32)
    y = np.array(df["label"].values, dtype=str)
    
    print("\n[2/3] Evaluating Model via 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    y_pred = cross_val_predict(model, X, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, digits=3, output_dict=True)
    
    print(f"  5-Fold Overall Accuracy: {acc:.1%}")
    print(f"  5-Fold Balanced Accuracy: {bal_acc:.1%}")
    print("\nConfusion Matrix (Rows=GT, Cols=Pred):")
    cm = confusion_matrix(y, y_pred, labels=["High", "Moderate", "Low"])
    print(cm)
    
    print("\n[3/3] Fitting Final Production Model on all validated data...")
    final_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X, y)
    
    artifact = {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": list(final_model.classes_),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "trained_samples": len(df),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(artifact, checkpoint_out)
    print(f"\nSUCCESS! Final model saved to: {checkpoint_out}")
    print(f"File size: {checkpoint_out.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    train_and_save()
