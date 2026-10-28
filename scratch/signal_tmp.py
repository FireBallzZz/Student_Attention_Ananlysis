"""FINAL: what observable actually separates the GT labels? (within-student centered)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

df = pd.read_csv("scratch/_clip4_ab_tmp.csv")
ALL = ['head_pitch_deg', 'head_yaw_deg', 'head_roll_deg', 'head_pose_confidence',
       'gaze_horizontal', 'gaze_vertical', 'torso_angle_deg', 'shoulder_tilt_deg',
       'movement_energy', 'bbox_area_ratio', 'detection_confidence', 'missing_face',
       'missing_head_pose', 'missing_gaze', 'missing_torso', 'is_distant']

df = df.sort_values(["gt_track_id", "frame_idx"]).reset_index(drop=True)
g = df["gt_track_id"].to_numpy(); y = df["label"].to_numpy().astype(str)

# within-student centering removes every static student/camera identity cue
d = df.copy()
for c in ALL:
    d[c] = d[c] - d.groupby("gt_track_id")[c].transform("mean")
# also remove absolute static cues entirely for a second view
BEHAV = ['head_pitch_deg', 'head_yaw_deg', 'head_roll_deg', 'head_pose_confidence',
         'gaze_horizontal', 'gaze_vertical', 'torso_angle_deg', 'shoulder_tilt_deg',
         'movement_energy', 'missing_face', 'missing_head_pose', 'missing_gaze',
         'missing_torso', 'is_distant']

print("=== mutual information with the GT label (within-student centered features) ===")
Xi = d[ALL].to_numpy(dtype=np.float32)
mi = mutual_info_classif(Xi, y, random_state=42)
for c, v in sorted(zip(ALL, mi), key=lambda t: -t[1]):
    print(f"  {c:22s} MI={v:.5f}")

print("\n=== LOSO with within-student centered features ===")
for name, cols in [("raw features", ALL),
                   ("within-student centered, all", ALL),
                   ("within-student centered, behavioural only", BEHAV)]:
    X = (df[cols].to_numpy(dtype=np.float32) if name == "raw features"
         else d[cols].to_numpy(dtype=np.float32))
    pred = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                             X, y, cv=LeaveOneOut())
    print(f"  {name:44s} acc={accuracy_score(y,pred):.4f} "
          f"balacc={balanced_accuracy_score(y,pred):.4f} macroF1={f1_score(y,pred,average='macro'):.4f}")

maj = df["label"].value_counts().idxmax()
print(f"\n  {'majority class baseline':44s} acc={accuracy_score(y, np.array([maj]*len(y))):.4f} "
      f"balacc=0.3333 macroF1=0.2846")

print("\n=== per-student dominant label achievable ceiling ===")
ref = df.groupby("gt_track_id")["label"].agg(lambda s: s.value_counts().idxmax())
mapped = df["gt_track_id"].map(ref)
print(f"  acc={accuracy_score(y, mapped):.4f} (identity oracle — NOT usable on unseen students)")
print("\n=== how often is the label stable within a student? ===")
print("  fraction of consecutive frame pairs with the same label:",
      round(float(df.groupby("gt_track_id")["label"].apply(
          lambda s: (s.to_numpy()[1:] == s.to_numpy()[:-1]).mean()).mean()), 4))
