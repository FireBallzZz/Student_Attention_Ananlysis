import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

from src.pipeline.video_pipeline import ClassroomAttentionPipeline
from src.tracking.tracker import ClassroomCensusTracker
from src.detection.pose_detector import YOLO11PoseDetector
from src.modalities.face_cropper import FaceCropper
from src.modalities.openseeface_tracker import OpenSeeFaceTracker
from src.modalities.facial_attention import ModalityExtractor
from src.features.feature_schema import create_feature_vector
from src.models.attention_scorer import AttentionScorer
from src.attention_drop.detector import AttentionDropDetector
from src.recommendations.recommendation_engine import RecommendationEngine

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    denom = float(areaA + areaB - inter)
    return inter / denom if denom > 0 else 0.0

def load_ground_truth(xml_path):
    print(f"Loading ground truth from {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # frame_idx -> list of {track_id, bbox, attention}
    gt_frames = defaultdict(list)
    gt_tracks = set()
    
    for track in root.findall(".//track"):
        tid = int(track.get("id"))
        gt_tracks.add(tid)
        for box in track.findall("box"):
            f = int(box.get("frame"))
            outside = int(box.get("outside", 0))
            if outside == 1:
                continue
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            
            att_elem = box.find(".//attribute[@name='Attention']")
            att_val = att_elem.text.strip() if att_elem is not None and att_elem.text else "Moderate"
            
            gt_frames[f].append({
                "track_id": tid,
                "bbox": (xtl, ytl, xbr, ybr),
                "attention": att_val
            })
            
    print(f"Loaded {len(gt_tracks)} ground truth student tracks across {len(gt_frames)} frames.")
    return gt_frames, gt_tracks

def run_evaluation():
    xml_path = Path("data/annotate for clip 4/annotations.xml")
    video_path = Path("data/clip 4.mp4")
    output_video_path = Path("web_app/static/outputs/annotated_clip_4.mp4")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    gt_frames, gt_tracks = load_ground_truth(xml_path)
    
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path.name}: {w}x{h} @ {fps:.2f} FPS ({total_frames} frames)...")
    
    # Initialize pipeline components
    detector = YOLO11PoseDetector(conf_threshold=0.15, imgsz=1280)
    tracker = ClassroomCensusTracker(census_window_frames=20, max_disappeared_frames=30)
    face_cropper = FaceCropper()
    osf_tracker = OpenSeeFaceTracker()
    mod_extractor = ModalityExtractor()
    attention_scorer = AttentionScorer()
    drop_detector = AttentionDropDetector(window_duration_sec=2.0, drop_magnitude_threshold=20.0)
    rec_engine = RecommendationEngine()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (1280, 720))
    
    # Evaluation metric accumulators
    total_gt_boxes = 0
    total_pred_boxes = 0
    total_true_positives_30 = 0
    total_true_positives_50 = 0
    matched_ious = []
    
    # Track discovery: which gt tracks were matched at least once
    gt_tracks_matched = set()
    
    # Classification comparison
    y_true_3class = []
    y_pred_3class = []
    
    y_true_binary = []
    y_pred_binary = []
    
    frame_idx = 0
    t0 = time.time()
    
    all_drop_events = []
    student_histories = defaultdict(list)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cur_frame_idx = frame_idx
        timestamp_sec = cur_frame_idx / fps
        frame_idx += 1
        
        # 1. Detection
        dets = detector.detect(frame)
        
        # 2. Tracking with Census & Previous Frame Reference
        tracked_students = tracker.update(frame, dets)
        
        # Modalities & Attention Scoring
        frame_predictions = []
        for s in tracked_students:
            crop_res = face_cropper.crop(frame, s.bbox, s.keypoints_xy, s.keypoints_conf)
            osf_res = osf_tracker.process_face(s.track_id, crop_res, s.keypoints_xy, s.keypoints_conf)
            mods = mod_extractor.extract(s, frame.shape)
            
            fv = create_feature_vector(
                session_id="clip_4",
                clip_id="clip_4",
                student=s,
                modalities=mods,
                frame_idx=cur_frame_idx,
                timestamp_sec=timestamp_sec,
                target_label=1
            )
            score_res = attention_scorer.score(fv)
            
            # Map model output category directly to evaluation classes
            if score_res.category == "Attentive":
                pred_label_3c = "High"
            elif score_res.category == "Inattentive":
                pred_label_3c = "Low"
            else:
                pred_label_3c = "Moderate"
                
            pred_binary = "High" if pred_label_3c == "High" else "Low"
            
            frame_predictions.append({
                "student": s,
                "score": score_res.score,
                "label_3c": pred_label_3c,
                "label_binary": pred_binary,
                "tracking_mode": s.tracking_mode
            })
            
            # Drop detection
            drop_ev = drop_detector.add_observation(
                track_id=s.track_id,
                frame_idx=cur_frame_idx,
                timestamp_sec=timestamp_sec,
                score=score_res.score,
                head_pitch_deg=osf_res.head_pitch,
                head_yaw_deg=osf_res.head_yaw,
                torso_angle_deg=mods.torso_angle_deg,
                movement_energy=mods.movement_energy,
                is_missing=not osf_res.is_valid
            )
            if drop_ev:
                rec = rec_engine.generate_recommendation(drop_ev)
                all_drop_events.append({
                    "track_id": drop_ev.track_id,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "frame_idx": cur_frame_idx,
                    "drop_magnitude": round(drop_ev.drop_magnitude, 1),
                    "rec": rec.teaching_recommendation
                })

        # --- EVALUATION MATCHING WITH GROUND TRUTH ---
        gts = gt_frames.get(cur_frame_idx, [])
        total_gt_boxes += len(gts)
        total_pred_boxes += len(frame_predictions)
        
        # Greedy bipartite matching on IoU
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        # Build cost/iou matrix
        if len(gts) > 0 and len(frame_predictions) > 0:
            iou_mat = np.zeros((len(gts), len(frame_predictions)), dtype=np.float32)
            for gi, gt in enumerate(gts):
                for pi, pred in enumerate(frame_predictions):
                    iou_mat[gi, pi] = compute_iou(gt["bbox"], pred["student"].bbox)
                    
            while True:
                max_iou = np.max(iou_mat)
                if max_iou < 0.20:
                    break
                gi, pi = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                
                gt = gts[gi]
                pred = frame_predictions[pi]
                
                matched_gt_indices.add(gi)
                matched_pred_indices.add(pi)
                matched_ious.append(float(max_iou))
                gt_tracks_matched.add(gt["track_id"])
                
                if max_iou >= 0.30:
                    total_true_positives_30 += 1
                if max_iou >= 0.50:
                    total_true_positives_50 += 1
                    
                # Classification pairing
                gt_att = gt["attention"] # 'High', 'Moderate', 'Low'
                pred_att_3c = pred["label_3c"]
                
                y_true_3class.append(gt_att)
                y_pred_3class.append(pred_att_3c)
                
                # Binary pairing: High vs (Moderate + Low)
                gt_bin = "High" if gt_att == "High" else "Low"
                y_true_binary.append(gt_bin)
                y_pred_binary.append(pred["label_binary"])
                
                iou_mat[gi, :] = -1.0
                iou_mat[:, pi] = -1.0

        # --- VISUAL RENDERING (Diagnostic Frame) ---
        disp_frame = frame.copy()
        
        # Draw Ground Truth in thin Cyan
        for gt in gts:
            gx1, gy1, gx2, gy2 = [int(v) for v in gt["bbox"]]
            cv2.rectangle(disp_frame, (gx1, gy1), (gx2, gy2), (200, 200, 0), 1)
            cv2.putText(disp_frame, f"GT#{gt['track_id']}:{gt['attention'][0]}", (gx1, max(12, gy1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)

        # Draw Predictions
        for p in frame_predictions:
            s = p["student"]
            px1, py1, px2, py2 = [int(v) for v in s.bbox]
            
            if p["label_3c"] == "High":
                color = (0, 220, 100) # Emerald
            elif p["label_3c"] == "Low":
                color = (60, 60, 240) # Red
            else:
                color = (0, 180, 240) # Amber
                
            cv2.rectangle(disp_frame, (px1, py1), (px2, py2), color, 2)
            mem_flag = " [M]" if p["tracking_mode"] == "COASTING" else ""
            badge = f"P#{s.track_id}:{p['score']:.0f}{mem_flag}"
            cv2.putText(disp_frame, badge, (px1 + 2, py2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        # Telemetry HUD
        cv2.rectangle(disp_frame, (0, 0), (w, 55), (15, 20, 30), -1)
        hud_l1 = f"Clip 4 Benchmark | Frame: {cur_frame_idx}/300 | Time: {timestamp_sec:.1f}s"
        hud_r1 = f"GT Students: {len(gts)} | Model Tracked: {len(frame_predictions)} | Census: {tracker.census_count}"
        cv2.putText(disp_frame, hud_l1, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 235, 245), 1)
        cv2.putText(disp_frame, hud_r1, (w - 480, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 240), 1)
        
        hud_l2 = f"Spatial IoU Recall: {total_true_positives_30 / max(1, total_gt_boxes):.1%} | Dynamics: {tracker.last_event or 'STABLE'}"
        cv2.putText(disp_frame, hud_l2, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 200), 1)

        out_frame = cv2.resize(disp_frame, (1280, 720))
        writer.write(out_frame)

    cap.release()
    writer.release()
    total_time = time.time() - t0
    
    # Compute Final Evaluation Metrics
    precision_30 = total_true_positives_30 / max(1, total_pred_boxes)
    recall_30 = total_true_positives_30 / max(1, total_gt_boxes)
    f1_det_30 = 2 * precision_30 * recall_30 / max(1e-6, (precision_30 + recall_30))
    
    precision_50 = total_true_positives_50 / max(1, total_pred_boxes)
    recall_50 = total_true_positives_50 / max(1, total_gt_boxes)
    f1_det_50 = 2 * precision_50 * recall_50 / max(1e-6, (precision_50 + recall_50))
    
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    
    # Classification Metrics (3-Class: High, Moderate, Low)
    labels_3c = ["High", "Moderate", "Low"]
    acc_3c = accuracy_score(y_true_3class, y_pred_3class) if y_true_3class else 0.0
    bal_acc_3c = balanced_accuracy_score(y_true_3class, y_pred_3class) if y_true_3class else 0.0
    macro_f1_3c = f1_score(y_true_3class, y_pred_3class, average="macro", labels=labels_3c) if y_true_3class else 0.0
    cm_3c = confusion_matrix(y_true_3class, y_pred_3class, labels=labels_3c) if y_true_3class else np.zeros((3, 3))
    report_3c = classification_report(y_true_3class, y_pred_3class, labels=labels_3c, output_dict=True) if y_true_3class else {}
    
    # Binary Metrics (High vs Low/Moderate)
    acc_bin = accuracy_score(y_true_binary, y_pred_binary) if y_true_binary else 0.0
    bal_acc_bin = balanced_accuracy_score(y_true_binary, y_pred_binary) if y_true_binary else 0.0
    f1_bin = f1_score(y_true_binary, y_pred_binary, pos_label="High") if y_true_binary else 0.0
    cm_bin = confusion_matrix(y_true_binary, y_pred_binary, labels=["High", "Low"]) if y_true_binary else np.zeros((2, 2))
    
    print("\n" + "=" * 80)
    print("EXPERIMENTAL EVALUATION RESULTS ON ANNOTATED CLIP 4")
    print("=" * 80)
    print(f"Total Video Frames Evaluated: {frame_idx} (Duration: {frame_idx/fps:.2f}s)")
    print(f"Processing Speed: {frame_idx / total_time:.1f} FPS (Total Time: {total_time:.1f}s)")
    print("\n--- 1. DETECTION & STUDENT CENSUS PERFORMANCE ---")
    print(f"Ground Truth Seated Students in Room: {len(gt_tracks)} (IDs 0..20)")
    print(f"Ground Truth Tracks Discovered & Matched: {len(gt_tracks_matched)} / {len(gt_tracks)} ({len(gt_tracks_matched)/len(gt_tracks):.1%})")
    print(f"Model Established Room Census: {tracker.census_count} students")
    print(f"Total Ground Truth Bounding Boxes: {total_gt_boxes}")
    print(f"Total Model Predicted Boxes: {total_pred_boxes}")
    print(f"Mean Bounding Box IoU: {mean_iou:.3f}")
    print(f"Detection @ IoU 0.30: Precision={precision_30:.1%}, Recall={recall_30:.1%}, F1={f1_det_30:.1%}")
    print(f"Detection @ IoU 0.50: Precision={precision_50:.1%}, Recall={recall_50:.1%}, F1={f1_det_50:.1%}")
    
    print("\n--- 2. ATTENTION PREDICTION ACCURACY (3-Class: High / Moderate / Low) ---")
    print(f"Total Matched Student-Frame Observations: {len(y_true_3class)}")
    print(f"3-Class Overall Accuracy: {acc_3c:.1%}")
    print(f"3-Class Balanced Accuracy: {bal_acc_3c:.1%}")
    print(f"3-Class Macro F1: {macro_f1_3c:.3f}")
    print("3-Class Confusion Matrix (Rows=GT, Cols=Pred):")
    print("                Pred High    Pred Moderate    Pred Low")
    print(f"GT High:        {cm_3c[0,0]:9d}    {cm_3c[0,1]:13d}    {cm_3c[0,2]:8d}")
    print(f"GT Moderate:    {cm_3c[1,0]:9d}    {cm_3c[1,1]:13d}    {cm_3c[1,2]:8d}")
    print(f"GT Low:         {cm_3c[2,0]:9d}    {cm_3c[2,1]:13d}    {cm_3c[2,2]:8d}")
    
    print("\nPer-Class Detailed Metrics:")
    for cls in labels_3c:
        if cls in report_3c:
            p = report_3c[cls]["precision"]
            r = report_3c[cls]["recall"]
            f = report_3c[cls]["f1-score"]
            supp = report_3c[cls]["support"]
            print(f"  Class {cls:8s}: Precision={p:.1%}, Recall={r:.1%}, F1={f:.3f} (Support: {supp})")
            
    print("\n--- 3. BINARY ATTENTION ACCURACY (High Focus vs Disengaged/Low) ---")
    print(f"Binary Accuracy: {acc_bin:.1%}")
    print(f"Binary Balanced Accuracy: {bal_acc_bin:.1%}")
    print(f"Binary F1-Score: {f1_bin:.3f}")
    print("Binary Confusion Matrix (Rows=GT, Cols=Pred):")
    print(f"GT High: {cm_bin[0,0]:5d} (TP), {cm_bin[0,1]:5d} (FN)")
    print(f"GT Low : {cm_bin[1,0]:5d} (FP), {cm_bin[1,1]:5d} (TN)")
    
    # Save full results json
    results_json = {
        "clip_name": "clip 4.mp4",
        "total_frames": frame_idx,
        "duration_sec": frame_idx / fps,
        "processing_fps": round(frame_idx / total_time, 1),
        "gt_students_count": len(gt_tracks),
        "gt_tracks_matched": len(gt_tracks_matched),
        "model_census_count": tracker.census_count,
        "detection_metrics": {
            "mean_iou": round(mean_iou, 3),
            "precision_iou30": round(precision_30 * 100, 2),
            "recall_iou30": round(recall_30 * 100, 2),
            "f1_iou30": round(f1_det_30 * 100, 2),
            "precision_iou50": round(precision_50 * 100, 2),
            "recall_iou50": round(recall_50 * 100, 2),
            "f1_iou50": round(f1_det_50 * 100, 2)
        },
        "classification_3class": {
            "overall_accuracy": round(acc_3c * 100, 2),
            "balanced_accuracy": round(bal_acc_3c * 100, 2),
            "macro_f1": round(macro_f1_3c, 3),
            "confusion_matrix": cm_3c.tolist(),
            "per_class": report_3c
        },
        "classification_binary": {
            "accuracy": round(acc_bin * 100, 2),
            "balanced_accuracy": round(bal_acc_bin * 100, 2),
            "f1_score": round(f1_bin, 3),
            "confusion_matrix": cm_bin.tolist()
        },
        "drop_events_count": len(all_drop_events),
        "output_video_path": str(output_video_path.resolve())
    }
    
    with open("reports/clip_4_accuracy_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print("\nSaved evaluation results to reports/clip_4_accuracy_evaluation.json")
    print(f"Rendered diagnostic video to: {output_video_path.resolve()}")

if __name__ == "__main__":
    run_evaluation()
