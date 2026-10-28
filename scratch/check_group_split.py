import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score

from src.features.feature_schema import FEATURE_COLUMNS

def test_group_kfold():
    # Load the extracted features from clip 4
    # Check if we saved them or re-extract
    from scratch.test_feature_discrimination import test_feature_discrimination
    
    # We can run GroupKFold using the 21 student track IDs
    # Let's inspect the distribution across students
    df = pd.read_csv("scratch/_clip4_features_tmp.csv") if Path("scratch/_clip4_features_tmp.csv").exists() else None
    if df is None:
        print("Features tmp not found, generating...")
        import scratch.test_feature_discrimination as tfd
        # Let's check how many samples per track
    
if __name__ == "__main__":
    test_group_kfold()
