import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("classroom_attention_research/checkpoints/yolo11n-pose.pt")

for vpath in ["data/pilot/high_engagement/high/view1.mp4", "data/pilot/low_engagement/low/view3.mp4"]:
    cap = cv2.VideoCapture(vpath)
    counts_def = []
    counts_imp = []
    all_boxes_def = []
    all_boxes_imp = []
    
    for f_idx in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx % 5 == 0:
            res_def = model.predict(frame, conf=0.35, imgsz=640, device="cuda:0", verbose=False)[0]
            res_imp = model.predict(frame, conf=0.15, imgsz=1280, device="cuda:0", verbose=False)[0]
            counts_def.append(len(res_def.boxes))
            counts_imp.append(len(res_imp.boxes))
    cap.release()
    print(f"File: {vpath}")
    print(f"  Counts default (conf=0.35, 640):  min={min(counts_def)}, max={max(counts_def)}, mean={np.mean(counts_def):.1f}")
    print(f"  Counts improved (conf=0.15, 1280): min={min(counts_imp)}, max={max(counts_imp)}, mean={np.mean(counts_imp):.1f}")
