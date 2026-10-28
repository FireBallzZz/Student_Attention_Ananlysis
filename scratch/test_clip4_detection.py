import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.detection.pose_detector import YOLO11PoseDetector

def test_detect():
    cap = cv2.VideoCapture("data/clip 4.mp4")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read frame")
        return
    
    print(f"Frame shape: {frame.shape}")
    detector = YOLO11PoseDetector(conf_threshold=0.15, imgsz=1280)
    dets = detector.detect(frame)
    print(f"Detections at conf=0.15, imgsz=1280: {len(dets)} detected")
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d.bbox
        print(f"  Det {i:2d}: ({x1:5.1f}, {y1:5.1f}, {x2:5.1f}, {y2:5.1f}) | W: {x2-x1:4.1f}, H: {y2-y1:4.1f} | Conf: {d.confidence:.2f}")

if __name__ == "__main__":
    test_detect()
