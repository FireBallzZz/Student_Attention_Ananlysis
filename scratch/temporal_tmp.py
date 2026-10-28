"""Temporary: test concrete accuracy-recovery options under honest grouped CV."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.detection.pose_detector import YOLO11PoseDetector
from src.tracking.tracker import ClassroomCensusTracker
from src.modalities.face_cropper import FaceCropper
from src.modalities.openseeface_tracker import OpenSeeFaceTracker
from src.modalities.facial_attention import ModalityExtractor
from src.features.feature_schema import FEATURE_COLUMNS, create_feature_vector
from scratch.evaluate_clip4_accuracy import compute_iou, load_ground_truth

CACHE = Path("scratch/_clip4_temporal_tmp.csv")
ALL = list(FEATURE_COLUMNS)


def signed_tilt(kxy, i, j):
    dx = float(kxy[j][0] - kxy[i][0]); dy = float(kxy[j][1] - kxy[i][1])
    if abs(dx) < 1e-3:
        return 0.0
    return float(np.degrees(np.arctan2(dy, abs(dx))))


def extract():
    if CACHE.exists():
        return pd.read_csv(CACHE)
    gt_frames, _ = load_ground_truth(Path("data/annotate for clip 4/annotations.xml"))
    cap = cv2.VideoCapture("data/clip 4.mp4")
    detector = YOLO11PoseDetector(conf_threshold=0.15, imgsz=1280)
    tracker = ClassroomCensusTracker()
    cropper = FaceCropper(); osf = OpenSeeFaceTracker(); mod = ModalityExtractor()
    prev_c = {}
    recs = []; fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur = fi; ts = cur / 29.97; fi += 1
        gts = gt_frames.get(cur, [])
        if not gts:
            continue
        students = tracker.update(frame, detector.detect(frame))
        for gt in gts:
            best, biou = None, 0.0
            for s in students:
                iou = compute_iou(gt["bbox"], s.bbox)
                if iou > biou:
                    biou, best = iou, s
            if best is None or biou < 0.30:
                continue
            cr = cropper.crop(frame, best.bbox, best.keypoints_xy, best.keypoints_conf)
            o = osf.process_face(best.track_id, cr, best.keypoints_xy, best.keypoints_conf)
            m = mod.extract(best, frame.shape)
            fv = create_feature_vector("clip_4", "clip_4", best, m, cur, ts, 1)
            d = {c: getattr(fv, c) for c in ALL}
            kxy, kcf = best.keypoints_xy, best.keypoints_conf
            d["head_roll_deg"] = signed_tilt(kxy, 2, 1) if (kcf[1] > .35 and kcf[2] > .35) else 0.0
            d["shoulder_tilt_deg"] = signed_tilt(kxy, 6, 5) if (kcf[5] > .35 and kcf[6] > .35) else 0.0
            cx = (best.bbox[0] + best.bbox[2]) / 2; cy = (best.bbox[1] + best.bbox[3]) / 2
            pc = prev_c.get(best.track_id)
            d["movement_energy"] = float(np.hypot(cx - pc[0], cy - pc[1])) if pc else 0.0
            prev_c[best.track_id] = (cx, cy)
            d.update({"label": gt["attention"], "gt_track_id": gt["track_id"], "frame_idx": cur})
            recs.append(d)
    cap.release()
    df = pd.DataFrame(recs); df.to_csv(CACHE, index=False)
    print("extracted", len(df)); return df


df = extract().reset_index(drop=True)
y = np.asarray(df["label"].to_numpy(), dtype=object).astype(str)
g = np.asarray(df["gt_track_id"].to_numpy())
maj = df["label"].value_counts().idxmax()
print(f"majority baseline: acc={accuracy_score(y, np.array([maj]*len(y))):.4f}")


def ev(cols, name, groups=g, cv=None):
    X = np.asarray(df[cols].to_numpy(), dtype=np.float32)
    mk = lambda: RandomForestClassifier(n_estimators=200, max_depth=12,
                                        class_weight="balanced", random_state=42, n_jobs=-1)
    p = cross_val_predict(mk(), X, y, cv=cv or GroupKFold(n_splits=5), groups=groups) \
        if groups is not None else cross_val_predict(mk(), X, y, cv=cv)
    print(f"  {name:56s} acc={accuracy_score(y,p):.4f} balacc={balanced_accuracy_score(y,p):.4f} "
          f"macroF1={f1_score(y,p,average='macro'):.4f}")
    return p


# --- 1. add per-track causal rolling features (past-only) ---
df = df.sort_values(["gt_track_id", "frame_idx"]).reset_index(drop=True)
grp = df.groupby("gt_track_id")
for col, win in [("head_pitch_deg", 30), ("head_yaw_deg", 30), ("torso_angle_deg", 60),
                 ("shoulder_tilt_deg", 60)]:
    df[f"{col}_roll_mean"] = grp[col].transform(lambda s: s.shift(1).rolling(win, min_periods=1).mean())
    df[f"{col}_roll_std"] = grp[col].transform(lambda s: s.shift(1).rolling(win, min_periods=1).std().fillna(0))
    df[f"{col}_delta"] = grp[col].diff().fillna(0)

# deviations from the student's OWN baseline (causal)
for col in ["head_pitch_deg", "head_yaw_deg", "torso_angle_deg"]:
    df[f"{col}_dev"] = df[col] - df[f"{col}_roll_mean"]

df = df.reset_index(drop=True)
y = np.asarray(df["label"].to_numpy(), dtype=object).astype(str)
g = np.asarray(df["gt_track_id"].to_numpy())

print("\n=== A) baseline (current feature set), GROUPED by student ===")
ev(ALL, "current 16 features")

print("\n=== B) + per-track causal rolling stats & deviations, GROUPED ===")
new = ALL + [c for c in df.columns if c.endswith(("_roll_mean", "_roll_std", "_delta", "_dev"))]
ev(new, "16 + rolling/rolling-std/delta/deviation features")

print("\n=== C) rolling features only (drop raw static geometry) ===")
rolling_only = [c for c in df.columns if c.endswith(("_roll_mean", "_roll_std", "_delta", "_dev"))] + \
                ["missing_face", "missing_head_pose", "missing_gaze", "missing_torso", "is_distant"]
ev(rolling_only, "rolling + missingness only")

print("\n=== D) per-frame behavioural only + deviations ===")
beh = ["head_pitch_deg", "head_yaw_deg", "head_roll_deg", "movement_energy",
       "head_pitch_deg_dev", "head_yaw_deg_dev", "torso_angle_deg_dev",
       "head_pitch_deg_delta", "head_yaw_deg_delta",
       "missing_face", "is_distant"]
ev(beh, "behavioural + deviations (no position proxies)")

print("\n=== E) per-frame + rolling, ROW-level (leaky, for reference) ===")
from sklearn.model_selection import StratifiedKFold
ev(new, "16 + rolling features, row-level", groups=None, cv=StratifiedKFold(5, shuffle=True, random_state=42))

print("\n=== F) honest POOLED (train on 16 clips, test on 5 clips of the SAME session set) ===")
# group by clip instead of student: unseen clip, same camera session
df["clip_id"] = df["frame_idx"] // 1  # placeholder, replaced below
