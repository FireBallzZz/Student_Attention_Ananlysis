import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from src.detection.pose_detector import YOLO11PoseDetector
from src.tracking.tracker import ClassroomCensusTracker
from src.modalities.face_cropper import FaceCropper
from src.modalities.openseeface_tracker import OpenSeeFaceTracker
from src.modalities.facial_attention import ModalityExtractor

def check_low_tracks():
    tree = ET.parse("data/annotate for clip 4/annotations.xml")
    root = tree.getroot()
    
    cap = cv2.VideoCapture("data/clip 4.mp4")
    ret, frame = cap.read()
    cap.release()
    
    detector = YOLO11PoseDetector(conf_threshold=0.15, imgsz=1280)
    tracker = ClassroomCensusTracker()
    face_cropper = FaceCropper()
    osf_tracker = OpenSeeFaceTracker()
    mod_extractor = ModalityExtractor()
    
    dets = detector.detect(frame)
    students = tracker.update(frame, dets)
    
    # Map gt tracks on frame 0
    gt_boxes = {}
    for track in root.findall(".//track"):
        tid = int(track.get("id"))
        box0 = track.find(".//box[@frame='0']")
        if box0 is not None:
            att = box0.find(".//attribute[@name='Attention']").text
            gt_boxes[tid] = {
                "bbox": (float(box0.get("xtl")), float(box0.get("ytl")), float(box0.get("xbr")), float(box0.get("ybr"))),
                "att": att
            }
            
    print("\nComparing GT tracks vs detected features on frame 0:")
    for gt_id, gt_data in sorted(gt_boxes.items()):
        gb = gt_data["bbox"]
        # Find closest student
        best_s = None
        best_iou = 0
        for s in students:
            from scratch.evaluate_clip4_accuracy import compute_iou
            iou = compute_iou(gb, s.bbox)
            if iou > best_iou:
                best_iou = iou
                best_s = s
        if best_s and best_iou > 0.3:
            crop_res = face_cropper.crop(frame, best_s.bbox, best_s.keypoints_xy, best_s.keypoints_conf)
            osf_res = osf_tracker.process_face(best_s.track_id, crop_res, best_s.keypoints_xy, best_s.keypoints_conf)
            mods = mod_extractor.extract(best_s, frame.shape)
            print(f"GT #{gt_id:2d} ({gt_data['att']:8s}) -> Pred #{best_s.track_id:2d} (IoU={best_iou:.2f}) | HeadPitch={osf_res.head_pitch:5.1f}°, HeadYaw={osf_res.head_yaw:5.1f}°, Torso={mods.torso_angle_deg:4.1f}°, Distant={mods.is_distant_student}")

if __name__ == "__main__":
    check_low_tracks()
