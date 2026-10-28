import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score

def test_unseen_student_split():
    df = pd.read_csv("scratch/_clip4_ab_tmp.csv")
    
    # Let's inspect class distribution per student
    print("Class distribution per student ID:")
    ct = pd.crosstab(df["gt_track_id"], df["label"])
    print(ct)
    
    # Split students:
    # Train students: e.g. 15 students (balanced High, Low, Moderate)
    # Test students: 6 students (completely held-out, unseen)
    test_students = [1, 2, 8, 14, 16, 20] # includes High, Moderate, Low students
    train_students = [tid for tid in df["gt_track_id"].unique() if tid not in test_students]
    
    df_train = df[df["gt_track_id"].isin(train_students)]
    df_test = df[df["gt_track_id"].isin(test_students)]
    
    print(f"\nTrain Set: {len(df_train)} samples across {len(train_students)} students ({train_students})")
    print(f"Test Set : {len(df_test)} samples across {len(test_students)} students ({test_students})")
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
    print("EVALUATION ON COMPLETELY UNSEEN STUDENTS:")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.1%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["High", "Moderate", "Low"]))

if __name__ == "__main__":
    test_unseen_student_split()
