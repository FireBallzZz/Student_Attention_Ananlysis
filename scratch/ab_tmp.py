"""Temporary: rigorous A/B of feature bugs under identical grouped folds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.detection.pose_detector import YOLO11PoseDetector
from src.tracking.tracker import ClassroomCensusTracker
from src.modalities.face_cropper import FaceCropper
from src.modalities.openseeface_tracker import OpenSeeFaceTracker
from src.modalities.facial_attention import ModalityExtractor
from src.features.feature_schema import FEATURE_COLUMNS, create_feature_vector
from scratch.evaluate_clip4_accuracy import compute_iou, load_ground_truth

CACHE = Path("scratch/_clip4_ab_tmp.csv")
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
            d["_roll_fixed"] = signed_tilt(kxy, 2, 1) if (kcf[1] > .35 and kcf[2] > .35) else 0.0
            d["_sh_fixed"] = signed_tilt(kxy, 6, 5) if (kcf[5] > .35 and kcf[6] > .35) else 0.0
            cx = (best.bbox[0] + best.bbox[2]) / 2; cy = (best.bbox[1] + best.bbox[3]) / 2
            pc = prev_c.get(best.track_id)
            d["_energy_fixed"] = float(np.hypot(cx - pc[0], cy - pc[1])) if pc else 0.0
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
print(f"majority baseline acc={accuracy_score(y, np.array([maj]*len(y))):.4f}")
print("students:", len(set(g)), "rows:", len(df))

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(np.zeros(len(y)), y, groups=g))
fold_of = np.zeros(len(y), dtype=int)
for k, (_, te) in enumerate(folds):
    fold_of[te] = k
print("fold sizes:", np.bincount(fold_of), "| students per fold:",
      [len(set(g[fold_of == k])) for k in range(5)])


def ev(cols, name, show_folds=True):
    X = np.asarray(df[cols].to_numpy(), dtype=np.float32)
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    acc = accuracy_score(y, pred); bal = balanced_accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    line = f"  {name:50s} acc={acc:.4f} balacc={bal:.4f} macroF1={f1:.4f}"
    if show_folds:
        per = [f"{accuracy_score(y[fold_of==k], pred[fold_of==k]):.3f}" for k in range(5)]
        line += "  folds=[" + ", ".join(per) + "]"
    print(line)
    return pred


print("\n=== RAW (as currently deployed) ===")
ev(ALL, "raw 16 features")

print("\n=== FIXED roll + shoulder + real energy ===")
fixed = [c for c in ALL if c not in ("head_roll_deg", "shoulder_tilt_deg", "movement_energy")]
fixed = fixed + ["_roll_fixed", "_sh_fixed", "_energy_fixed"]
ev(fixed, "fixed roll/shoulder/energy (16)")

print("\n=== FIXED, and drop camera-position proxies ===")
nopos = [c for c in fixed if c not in ("bbox_area_ratio", "detection_confidence")]
ev(nopos, "fixed minus area_ratio/confidence")

print("\n=== FIXED, minus position AND static identity geometry ===")
noid = [c for c in nopos if c not in ("_roll_fixed", "_sh_fixed")]
ev(noid, "fixed minus roll/shoulder")

print("\n=== sanity: can features identify the student? ===")
Xid = np.asarray(df[["_roll_fixed", "_sh_fixed", "torso_angle_deg", "bbox_area_ratio",
                      "detection_confidence", "head_yaw_deg"]].to_numpy(), dtype=np.float32)
pid = np.empty(len(y), dtype=object)
for tr, te in folds:
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xid[tr], g[tr])
    pid[te] = clf.predict(Xid[te])
print(f"  student-ID accuracy (grouped, unseen students): {accuracy_score(g.astype(str), pid):.4f}"
      f"  (chance {1/len(set(g)):.4f})")
