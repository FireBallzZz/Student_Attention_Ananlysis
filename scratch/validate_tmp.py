"""Temporary: validate the temporal-baseline gain is not an artifact.

Checks:
 1. evaluate only frames after a warm-up (>=30 frames of history) 
 2. evaluate the "predict each student's own dominant label" reference
 3. leave-one-student-out (LOSO) instead of GroupKFold, for a stricter check
 4. permutation control: shuffle labels within student, re-run
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

df = pd.read_csv("scratch/_clip4_ab_tmp.csv")
ALL = ['head_pitch_deg', 'head_yaw_deg', 'head_roll_deg', 'head_pose_confidence',
       'gaze_horizontal', 'gaze_vertical', 'torso_angle_deg', 'shoulder_tilt_deg',
       'movement_energy', 'bbox_area_ratio', 'detection_confidence', 'missing_face',
       'missing_head_pose', 'missing_gaze', 'missing_torso', 'is_distant']

df = df.sort_values(["gt_track_id", "frame_idx"]).reset_index(drop=True)
grp = df.groupby("gt_track_id")
for col, win in [("head_pitch_deg", 30), ("head_yaw_deg", 30), ("torso_angle_deg", 60)]:
    df[f"{col}_rm"] = grp[col].transform(lambda s: s.shift(1).rolling(win, min_periods=1).mean())
    df[f"{col}_rs"] = grp[col].transform(lambda s: s.shift(1).rolling(win, min_periods=1).std()).fillna(0)
    df[f"{col}_dev"] = df[col] - df[f"{col}_rm"]
    df[f"{col}_devabs"] = df[f"{col}_dev"].abs()
df["pitch_range"] = grp["head_pitch_deg"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).max() - s.shift(1).rolling(30, min_periods=1).min())

y = np.asarray(df["label"].to_numpy(), dtype=object).astype(str)
g = np.asarray(df["gt_track_id"].to_numpy())
fi = np.asarray(df["frame_idx"].to_numpy())

ROLL = [c for c in df.columns if c.endswith(("_rm", "_rs", "_dev", "_devabs"))] + ["pitch_range"]
maj = df["label"].value_counts().idxmax()
print("majority:", round(accuracy_score(y, np.array([maj]*len(y))), 4))

# reference: per-student dominant label
ref = np.zeros(len(y), dtype=object)
for tid, gg in df.groupby("gt_track_id"):
    ref[gg.index] = gg["label"].value_counts().idxmax()
print("per-student dominant-label reference:", round(accuracy_score(y, ref), 4))


def loso(cols, mask=None, name="", shuffle=False):
    X = np.asarray(df[cols].to_numpy(), dtype=np.float32)
    yy = y.copy()
    if shuffle:
        rng = np.random.default_rng(0)
        yy = pd.Series(y).groupby(g).transform(lambda s: rng.permutation(s.to_numpy())).to_numpy()
    pred = np.empty(len(y), dtype=object)
    for tid in np.unique(g):
        te = g == tid
        tr = ~te
        if mask is not None:
            tr = tr & mask
            te = te & mask
        clf = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
        clf.fit(X[tr], yy[tr])
        pred[te] = clf.predict(X[te])
    sel = np.ones(len(y), bool) if mask is None else mask
    print(f"  {name:52s} acc={accuracy_score(yy[sel],pred[sel]):.4f} "
          f"balacc={balanced_accuracy_score(yy[sel],pred[sel]):.4f} "
          f"macroF1={f1_score(yy[sel],pred[sel],average='macro'):.4f}  (n={sel.sum()})")
    return pred


print("\n=== LOSO (leave-one-student-out) ===")
loso(ALL, None, "raw 16 features")
loso(ALL + ROLL, None, "raw 16 + temporal-baseline features")
loso(ROLL + ["missing_face", "is_distant"], None, "temporal-baseline + missingness only")

print("\n=== LOSO restricted to frames with >=30 frames of history ===")
warm = fi >= 30
loso(ALL, warm, "raw 16 features (warm frames only)")
loso(ALL + ROLL, warm, "raw 16 + temporal (warm frames only)")
loso(ROLL + ["missing_face", "is_distant"], warm, "temporal only (warm frames only)")

print("\n=== PERMUTATION CONTROL (labels shuffled within student) ===")
loso(ALL + ROLL, None, "raw+temporal, SHUFFLED labels", shuffle=True)
