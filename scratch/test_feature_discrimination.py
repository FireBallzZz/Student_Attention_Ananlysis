import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, accuracy_score

from src.detection.pose_detector import YOLO11PoseDetector
from src.tracking.tracker import ClassroomCensusTracker
from src.modalities.face_cropper import FaceCropper
from src.modalities.openseeface_tracker import OpenSeeFaceTracker
from src.modalities.facial_attention import ModalityExtractor
from src.features.feature_schema import create_feature_vector
from scratch.evaluate_clip4_accuracy import compute_iou, load_ground_truth

def test_feature_discrimination():
    xml_path = Path("data/annotate for clip 4/annotations.xml")
    video_path = Path("data/clip 4.mp4")
    
    gt_frames, _ = load_ground_truth(xml_path)
    cap = cv2.VideoCapture(str(video_path))
    
    detector = YOLO11PoseDetector(conf_threshold=0.15, imgsz=1280)
    tracker = ClassroomCensusTracker()
    face_cropper = FaceCropper()
    osf_tracker = OpenSeeFaceTracker()
    mod_extractor = ModalityExtractor()
    
    records = []
    frame_idx = 0
    
    print("Extracting features across all 300 frames of Clip 4...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur_f = frame_idx
        frame_idx += 1
        
        gts = gt_frames.get(cur_f, [])
        if not gts:
            continue
            
        dets = detector.detect(frame)
        students = tracker.update(frame, dets)
        
        # Match with gt
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
                
                records.append({
                    "frame": cur_f,
                    "gt_track_id": gt["track_id"],
                    "pred_track_id": best_s.track_id,
                    "label": gt["attention"],
                    "head_pitch": osf_res.head_pitch,
                    "head_yaw": abs(osf_res.head_yaw),
                    "head_roll": osf_res.head_roll,
                    "torso_angle": mods.torso_angle_deg,
                    "movement_energy": mods.movement_energy,
                    "face_valid": 1.0 if osf_res.is_valid else 0.0,
                    "area_ratio": best_s.area_ratio,
                    "is_distant": 1.0 if mods.is_distant_student else 0.0
                })
                
    cap.release()
    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} samples across features.")
    print("Label counts:\n", df["label"].value_counts())
    
    feature_cols = ["head_pitch", "head_yaw", "head_roll", "torso_angle", "movement_energy", "face_valid", "area_ratio", "is_distant"]
    
    # Feature means by class
    print("\nMean Feature Values by Ground Truth Label:")
    print(df.groupby("label")[feature_cols].mean().to_string())
    
    # Evaluate 3-Class Cross-Validated Random Forest with Class Balancing
    X = np.array(df[feature_cols].values, dtype=np.float32)
    y = np.array(df["label"].values, dtype=str)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    
    print("\n" + "=" * 60)
    print("5-FOLD CROSS VALIDATION WITH CLASS WEIGHT BALANCING:")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy_score(y, y_pred):.1%}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y, y_pred):.1%}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, digits=3))
    print("Confusion Matrix (High, Low, Moderate):")
    print(confusion_matrix(y, y_pred, labels=["High", "Low", "Moderate"]))

if __name__ == "__main__":
    test_feature_discrimination()
