"""Temporary: isolate WHY GroupKFold gave 0.96 while LOSO gave 0.49."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

df = pd.read_csv("scratch/_clip4_ab_tmp.csv").reset_index(drop=True)
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
df["pitch_range"] = grp["head_pitch_deg"].transform(
    lambda s: s.shift(1).rolling(30, min_periods=1).max() - s.shift(1).rolling(30, min_periods=1).min())

ROLL = [c for c in df.columns if c.endswith(("_rm", "_rs", "_dev", "_devabs"))] + ["pitch_range"]
y = np.asarray(df["label"].to_numpy(), dtype=object).astype(str)
g = np.asarray(df["gt_track_id"].to_numpy())

# fixed versions of the buggy columns
df_fixed = df.copy()
df_fixed["head_roll_deg"] = df["_roll_fixed"]
df_fixed["shoulder_tilt_deg"] = df["_sh_fixed"]
df_fixed["movement_energy"] = df["_energy_fixed"]


def gkf_eval(d, cols, name):
    X = np.asarray(d[cols].to_numpy(), dtype=np.float32)
    p = cross_val_predict(RandomForestClassifier(n_estimators=200, max_depth=12,
                                                  class_weight="balanced", random_state=42, n_jobs=-1),
                          X, y, cv=GroupKFold(n_splits=5), groups=g)
    print(f"  GroupKFold {name:44s} acc={accuracy_score(y,p):.4f} "
          f"balacc={balanced_accuracy_score(y,p):.4f} macroF1={f1_score(y,p,average='macro'):.4f}")


def loso_eval(d, cols, name):
    X = np.asarray(d[cols].to_numpy(), dtype=np.float32)
    pred = np.empty(len(y), dtype=object)
    for tid in np.unique(g):
        te = g == tid
        clf = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced",
                                     random_state=42, n_jobs=-1).fit(X[~te], y[~te])
        pred[te] = clf.predict(X[te])
    print(f"  LOSO       {name:44s} acc={accuracy_score(y,pred):.4f} "
          f"balacc={balanced_accuracy_score(y,pred):.4f} macroF1={f1_score(y,pred,average='macro'):.4f}")


print("=== RAW columns ===")
gkf_eval(df, ALL, "raw 16")
loso_eval(df, ALL, "raw 16")
gkf_eval(df, ALL + ROLL, "raw 16 + rolling")
loso_eval(df, ALL + ROLL, "raw 16 + rolling")

print("\n=== FIXED roll/shoulder/energy ===")
gkf_eval(df_fixed, ALL, "fixed 16")
loso_eval(df_fixed, ALL, "fixed 16")
gkf_eval(df_fixed, ALL + ROLL, "fixed 16 + rolling")
loso_eval(df_fixed, ALL + ROLL, "fixed 16 + rolling")

print("\n=== sanity: distribution of a rolling feature vs label ===")
print(df.groupby("label")[["head_pitch_deg_rm", "head_pitch_deg_devabs", "pitch_range"]].mean().round(3).to_string())
print("\n=== per-student mean of head_pitch_deg_devabs (should separate students) ===")
print(df.groupby("gt_track_id")["head_pitch_deg_devabs"].mean().round(3).to_string())
