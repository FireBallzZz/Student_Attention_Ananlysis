import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class RobustStudentTracker:
    def __init__(self, conf_threshold=0.18, imgsz=1280, max_disappeared=20, iou_dedup_thresh=0.50):
        self.model = YOLO("classroom_attention_research/checkpoints/yolo11n-pose.pt")
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.max_disappeared = max_disappeared
        self.iou_dedup_thresh = iou_dedup_thresh
        
        self.next_id = 1
        self.tracks = {} # id -> track dict
        self.census_completed = False
        self.initial_census_count = 0

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    def update(self, frame, frame_idx):
        h, w = frame.shape[:2]
        
        # 1. Sensitive Detection
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device="cuda:0",
            verbose=False
        )[0]

        raw_boxes = []
        raw_confs = []
        raw_kpts = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            kpts = results.keypoints.data.cpu().numpy() if results.keypoints is not None else None

            # Spatial De-duplication on raw detections
            keep = []
            for i in range(len(boxes)):
                is_dup = False
                for j in keep:
                    if self._compute_iou(boxes[i], boxes[j]) > self.iou_dedup_thresh:
                        is_dup = True
                        break
                if not is_dup:
                    keep.append(i)

            for idx in keep:
                raw_boxes.append(boxes[idx].tolist())
                raw_confs.append(float(confs[idx]))
                raw_kpts.append(kpts[idx] if kpts is not None else np.zeros((17, 3)))

        # 2. Match with Existing Tracks
        matched_track_ids = set()
        unmatched_dets = list(range(len(raw_boxes)))

        if len(self.tracks) > 0 and len(raw_boxes) > 0:
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(raw_boxes)))

            for i, tid in enumerate(track_ids):
                for j, dbox in enumerate(raw_boxes):
                    iou_matrix[i, j] = self._compute_iou(self.tracks[tid]['bbox'], dbox)

            # Greedy matching by highest IoU
            while True:
                max_val = np.max(iou_matrix)
                if max_val < 0.20: # Match threshold
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]
                
                # Update track with new detection
                self.tracks[tid]['bbox'] = raw_boxes[j]
                self.tracks[tid]['conf'] = raw_confs[j]
                self.tracks[tid]['kpts'] = raw_kpts[j]
                self.tracks[tid]['disappeared'] = 0
                self.tracks[tid]['status'] = 'ACTIVE'
                self.tracks[tid]['last_seen'] = frame_idx
                
                matched_track_ids.add(tid)
                if j in unmatched_dets:
                    unmatched_dets.remove(j)

                # Invalidate matched row and column
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # 3. Handle Unmatched Existing Tracks (Reference to Previous Frame)
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self.tracks[tid]['disappeared'] += 1
                box = self.tracks[tid]['bbox']
                
                # Check if moved near camera boundary (went out from camera)
                is_near_edge = (box[0] < 15 or box[1] < 15 or box[2] > w - 15 or box[3] > h - 15)
                
                if is_near_edge and self.tracks[tid]['disappeared'] > 5:
                    self.tracks[tid]['status'] = 'EXITED'
                elif self.tracks[tid]['disappeared'] > self.max_disappeared:
                    self.tracks[tid]['status'] = 'VANISHED'
                else:
                    # Persist box from previous frame
                    self.tracks[tid]['status'] = 'COASTING'

        # Purge vanished or permanently exited tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['status'] in ['VANISHED', 'EXITED'] and self.tracks[tid]['disappeared'] > self.max_disappeared:
                del self.tracks[tid]

        # 4. Register New Tracks from Unmatched Detections
        for j in unmatched_dets:
            # Check de-duplication with any active track
            is_dup = False
            for tid, tdata in self.tracks.items():
                if tdata['status'] in ['ACTIVE', 'COASTING']:
                    if self._compute_iou(tdata['bbox'], raw_boxes[j]) > self.iou_dedup_thresh:
                        is_dup = True
                        break
            if not is_dup:
                self.tracks[self.next_id] = {
                    'track_id': self.next_id,
                    'bbox': raw_boxes[j],
                    'conf': raw_confs[j],
                    'kpts': raw_kpts[j],
                    'disappeared': 0,
                    'status': 'ACTIVE',
                    'first_seen': frame_idx,
                    'last_seen': frame_idx
                }
                self.next_id += 1

        # Establish Initial Census Count (after first 10 frames)
        if not self.census_completed and frame_idx >= 10:
            active_count = len([t for t in self.tracks.values() if t['status'] in ['ACTIVE', 'COASTING']])
            self.initial_census_count = max(self.initial_census_count, active_count)
            self.census_completed = True

        visible_students = [t for t in self.tracks.values() if t['status'] in ['ACTIVE', 'COASTING']]
        return visible_students

# Test on View 3 (where 7-9 students are present)
tracker = RobustStudentTracker(conf_threshold=0.18, imgsz=1280)
cap = cv2.VideoCapture("data/pilot/low_engagement/low/view3.mp4")

counts = []
for f_idx in range(40):
    ret, frame = cap.read()
    if not ret:
        break
    visible = tracker.update(frame, f_idx)
    counts.append(len(visible))
cap.release()

print(f"Robust Tracker on View 3:")
print(f"  Frame counts (0 to 40): {counts[:15]}...")
print(f"  Min visible: {min(counts[5:])}, Max visible: {max(counts[5:])}, Stable Mean: {np.mean(counts[5:]):.1f}")
print(f"  Initial census count established: {tracker.initial_census_count}")
