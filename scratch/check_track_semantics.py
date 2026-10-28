import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import xml.etree.ElementTree as ET
import cv2
import numpy as np
from src.detection.pose_detector import YOLO11PoseDetector
from src.tracking.tracker import ClassroomCensusTracker
from src.modalities.face_cropper import FaceCropper
from src.modalities.openseeface_tracker import OpenSeeFaceTracker
from src.modalities.facial_attention import ModalityExtractor
from src.features.feature_schema import create_feature_vector

def analyze_track_modalities():
    tree = ET.parse("data/annotate for clip 4/annotations.xml")
    root = tree.getroot()
    
    # Check gt labels for each track
    gt_track_labels = {}
    for track in root.findall(".//track"):
        tid = int(track.get("id"))
        labels = [b.find(".//attribute[@name='Attention']").text for b in track.findall("box") if b.find(".//attribute[@name='Attention']") is not None]
        top_label = max(set(labels), key=labels.count) if labels else "Unknown"
        gt_track_labels[tid] = top_label
        
    print("Ground truth track predominant labels:")
    for tid, l in sorted(gt_track_labels.items()):
        print(f"  GT Track {tid:2d}: {l}")

if __name__ == "__main__":
    analyze_track_modalities()
