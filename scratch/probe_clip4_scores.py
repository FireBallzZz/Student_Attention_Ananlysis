import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from src.detection.pose_detector import YOLO11PoseDetector
from src.tracking.tracker import ClassroomCensusTracker
from src.modalities.face_cropper import FaceCropper
from src.modalities.openseeface_tracker import OpenSeeFaceTracker
from src.modalities.facial_attention import ModalityExtractor
from src.features.feature_schema import create_feature_vector
from src.models.attention_scorer import AttentionScorer

def probe():
    cap = cv2.VideoCapture("data/clip 4.mp4")
    detector = YOLO11PoseDetector(conf_threshold=0.15, imgsz=1280)
    tracker = ClassroomCensusTracker()
    face_cropper = FaceCropper()
    osf_tracker = OpenSeeFaceTracker()
    mod_extractor = ModalityExtractor()
    attention_scorer = AttentionScorer()
    
    scores = []
    uncertains = []
    distants = []
    head_pitches = []
    
    for f_idx in range(60): # 2 seconds
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        students = tracker.update(frame, dets)
        for s in students:
            crop_res = face_cropper.crop(frame, s.bbox, s.keypoints_xy, s.keypoints_conf)
            osf_res = osf_tracker.process_face(s.track_id, crop_res, s.keypoints_xy, s.keypoints_conf)
            mods = mod_extractor.extract(s, frame.shape)
            fv = create_feature_vector("s", "c", s, mods, f_idx, f_idx/30.0, 1)
            score_res = attention_scorer.score(fv)
            scores.append(score_res.score)
            uncertains.append(score_res.is_uncertain_due_to_missingness)
            distants.append(mods.is_distant_student)
            head_pitches.append(osf_res.head_pitch)
            
    cap.release()
    print("Scores summary over first 60 frames:")
    print(f"  Min score: {np.min(scores):.2f}, Max score: {np.max(scores):.2f}, Mean score: {np.mean(scores):.2f}, Median: {np.median(scores):.2f}")
    print(f"  Uncertain ratio: {np.mean(uncertains):.1%}")
    print(f"  Distant student ratio: {np.mean(distants):.1%}")
    print(f"  OSF Valid Faces ratio: {np.mean([p != 0.0 for p in head_pitches]):.1%}")
    print(f"  Score percentiles: 10th={np.percentile(scores, 10):.1f}, 25th={np.percentile(scores, 25):.1f}, 50th={np.percentile(scores, 50):.1f}, 75th={np.percentile(scores, 75):.1f}, 90th={np.percentile(scores, 90):.1f}")

if __name__ == "__main__":
    probe()
