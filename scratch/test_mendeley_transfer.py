import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score

def test_mendeley_to_clip4_transfer():
    mendeley_path = Path("experiments/dataset_pilot_001/data/mendeley/attention_detection_dataset_v3.csv")
    clip4_path = Path("scratch/_clip4_ab_tmp.csv")
    
    df_m = pd.read_csv(mendeley_path)
    df_c = pd.read_csv(clip4_path)
    
    print(f"Mendeley samples: {len(df_m)}")
    print(f"Clip 4 samples  : {len(df_c)}")
    
    # In Mendeley: label 0 = Attentive (High), 1 = Inattentive (Low)
    # Common features available in both:
    # head_pitch, head_yaw, head_roll
    # Let's inspect Mendeley head angles vs Clip 4 head angles
    print("\nMendeley Head Angles (Mean):")
    print(df_m.groupby("label")[["head_pitch", "head_yaw", "head_roll"]].mean())
    
    print("\nClip 4 Head Angles (Mean):")
    print(df_c.groupby("label")[["head_pitch_deg", "head_yaw_deg", "head_roll_deg", "torso_angle_deg"]].mean())

if __name__ == "__main__":
    test_mendeley_to_clip4_transfer()
