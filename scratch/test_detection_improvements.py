import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

video_path = "data/pilot/high_engagement/high/view1.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    h, w = frame.shape[:2]
    print(f"Frame shape: {w}x{h}")
    model = YOLO("classroom_attention_research/checkpoints/yolo11n-pose.pt")

    # Current settings: conf=0.35, imgsz=default (640)
    res_default = model.predict(frame, conf=0.35, device="cuda:0", verbose=False)[0]
    print(f"Current default detection (conf=0.35, imgsz=640): {len(res_default.boxes)} students detected")

    # Improved settings 1: conf=0.20, imgsz=1280
    res_improved1 = model.predict(frame, conf=0.20, imgsz=1280, device="cuda:0", verbose=False)[0]
    print(f"Improved detection 1 (conf=0.20, imgsz=1280): {len(res_improved1.boxes)} students detected")

    # Improved settings 2: conf=0.15, imgsz=1280
    res_improved2 = model.predict(frame, conf=0.15, imgsz=1280, device="cuda:0", verbose=False)[0]
    print(f"Improved detection 2 (conf=0.15, imgsz=1280): {len(res_improved2.boxes)} students detected")

    # Let's inspect view3 as well (side angle)
    cap3 = cv2.VideoCapture("data/pilot/low_engagement/low/view3.mp4")
    ret3, frame3 = cap3.read()
    cap3.release()
    if ret3:
        res3_def = model.predict(frame3, conf=0.35, device="cuda:0", verbose=False)[0]
        res3_imp = model.predict(frame3, conf=0.20, imgsz=1280, device="cuda:0", verbose=False)[0]
        print(f"\nView 3 Default (conf=0.35, imgsz=640): {len(res3_def.boxes)} students")
        print(f"View 3 Improved (conf=0.20, imgsz=1280): {len(res3_imp.boxes)} students")
