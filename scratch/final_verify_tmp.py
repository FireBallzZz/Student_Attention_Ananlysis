"""FINAL consolidated verification (temporary). Careful index alignment throughout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

df = pd.read_csv("scratch/_clip4_ab_tmp.csv")
ALL = ['head_pitch_deg', 'head_yaw_deg', 'head_roll_deg', 'head_pose_confidence',
       'gaze_horizontal', 'gaze_vertical', 'torso_angle_deg', 'shoulder_tilt_deg',
       'movement_energy', 'bbox_area_ratio', 'detection_confidence', 'missing_face',
       'missing_head_pose', 'missing_gaze', 'missing_torso', 'is_distant']

df = df.sort_values(["gt_track_id", "frame_idx"]).reset_index(drop=True)
g = df["gt_track_id"].to_numpy()
y = df["label"].to_numpy().astype(str)
print(f"rows={len(df)} students={len(set(g))} class dist={df['label'].value_counts().to_dict()}")

maj = df["label"].value_counts().idxmax()
print(f"[baseline] majority-class acc={accuracy_score(y, np.array([maj]*len(y))):.4f} balacc=0.3333")

df_fixed = df.copy()
df_fixed["head_roll_deg"] = df["_roll_fixed"]
df_fixed["shoulder_tilt_deg"] = df["_sh_fixed"]
df_fixed["movement_energy"] = df["_energy_fixed"]

grp = df.groupby("gt_track_id")
for col, win in [("head_pitch_deg", 30), ("head_yaw_deg", 30), ("torso_angle_deg", 60)]:
    df[f"{col}_rm"] = grp[col].transform(lambda s: s.shift(1).rolling(win, min_periods=1).mean())
    df[f"{col}_rs"] = grp[col].transform(lambda s: s.shift(1).rolling(win, min_periods=1).std()).fillna(0)
    df[f"{col}_dev"] = df[col] - df[f"{col}_rm"]
df["pitch_range"] = grp["head_pitch_deg"].transform(
    lambda s: s.shift(1).rolling(30, min_periods=1).max() - s.shift(1).rolling(30, min_periods=1).min())
ROLL = [c for c in df.columns if c.endswith(("_rm", "_rs", "_dev"))] + ["pitch_range"]

df = df.reset_index(drop=True)
g = df["gt_track_id"].to_numpy(); y = df["label"].to_numpy().astype(str)


def gkf(d, cols, name, yy=None):
    yy = y if yy is None else yy
    X = np.asarray(d[cols].to_numpy(), dtype=np.float32)
    p = cross_val_predict(RandomForestClassifier(n_estimators=200, max_depth=12,
                                                  class_weight="balanced", random_state=42, n_jobs=-1),
                          X, yy, cv=GroupKFold(n_splits=5), groups=g)
    print(f"  GroupKFold(unseen student) {name:38s} acc={accuracy_score(yy,p):.4f} "
          f"balacc={balanced_accuracy_score(yy,p):.4f} macroF1={f1_score(yy,p,average='macro'):.4f}")


def loso(d, cols, name, yy=None):
    yy = y if yy is None else yy
    X = np.asarray(d[cols].to_numpy(), dtype=np.float32)
    pred = np.empty(len(yy), dtype=object)
    for tid in np.unique(g):
        te = g == tid
        clf = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight="balanced",
                                     random_state=42, n_jobs=-1).fit(X[~te], yy[~te])
        pred[te] = clf.predict(X[te])
    print(f"  LOSO(unseen student)         {name:38s} acc={accuracy_score(yy,pred):.4f} "
          f"balacc={balanced_accuracy_score(yy,pred):.4f} macroF1={f1_score(yy,pred,average='macro'):.4f}")


def rowlvl(d, cols, name):
    X = np.asarray(d[cols].to_numpy(), dtype=np.float32)
    p = cross_val_predict(RandomForestClassifier(n_estimators=200, max_depth=12,
                                                  class_weight="balanced", random_state=42, n_jobs=-1),
                          X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"  ROW-LEVEL (leaky)            {name:38s} acc={accuracy_score(y,p):.4f} "
          f"balacc={balanced_accuracy_score(y,p):.4f} macroF1={f1_score(y,p,average='macro'):.4f}")


print("\n=== deployed feature set ===")
gkf(df, ALL, "raw 16"); loso(df, ALL, "raw 16"); rowlvl(df, ALL, "raw 16")
print("\n=== bug-fixed roll/shoulder/energy ===")
gkf(df_fixed, ALL, "fixed 16"); loso(df_fixed, ALL, "fixed 16")
print("\n=== + causal per-student temporal baselines ===")
gkf(df, ALL + ROLL, "raw+rolling"); loso(df, ALL + ROLL, "raw+rolling")
print("\n=== permutation control: labels shuffled WITHIN student ===")
rng = np.random.default_rng(0)
yshuf = pd.Series(y).groupby(g).transform(lambda s: rng.permutation(s.to_numpy())).to_numpy()
gkf(df, ALL + ROLL, "raw+rolling, SHUFFLED labels", yy=yshuf)
loso(df, ALL + ROLL, "raw+rolling, SHUFFLED labels", yy=yshuf)

print("\n=== identity memorisation check (can features identify the student?) ===")
Xid = np.asarray(df[["bbox_area_ratio", "detection_confidence", "torso_angle_deg",
                      "shoulder_tilt_deg", "head_yaw_deg"]].to_numpy(), dtype=np.float32)
pid = cross_val_predict(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                         Xid, g.astype(str), cv=GroupKFold(5), groups=g)
print(f"  student-ID acc on UNSEEN students = {accuracy_score(g.astype(str), pid):.4f} "
      f"(chance {1/len(set(g)):.4f})")
# same-video holdout of frames (students seen in train)
from sklearn.model_selection import train_test_split
tr, te = train_test_split(np.arange(len(g)), test_size=0.3, random_state=42, stratify=g)
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xid[tr], g[tr])
print(f"  student-ID acc on HELD-OUT FRAMES of same students = {clf.score(Xid[te], g[te]):.4f} "
      f"(chance {1/len(set(g)):.4f})")
