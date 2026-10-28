import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score

def test_temporal_split():
    df = pd.read_csv("scratch/_clip4_ab_tmp.csv")
    
    # Split by time:
    # Train: Frames 0 to 200 (66.7% of the video)
    # Test : Frames 201 to 299 (33.3% of the video - UNSEEN FUTURE)
    df_train = df[df["frame_idx"] <= 200]
    df_test = df[df["frame_idx"] > 200]
    
    print(f"Train Set (Frames 0-200): {len(df_train)} samples")
    print(f"Test Set (Frames 201-299): {len(df_test)} samples")
    print("Train label distribution:\n", df_train["label"].value_counts())
    print("Test label distribution:\n", df_test["label"].value_counts())
    
    feature_cols = ["head_pitch_deg", "head_yaw_deg", "head_roll_deg", "torso_angle_deg", "movement_energy", "bbox_area_ratio"]
    
    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    
    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("\n" + "=" * 60)
    print("TEMPORAL SPLIT EVALUATION (UNSEEN FUTURE FRAMES 201-299):")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.1%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["High", "Moderate", "Low"]))

if __name__ == "__main__":
    test_temporal_split()
