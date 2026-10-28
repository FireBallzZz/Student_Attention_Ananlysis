import pandas as pd
import numpy as np

csv_path = "experiments/dataset_pilot_001/data/mendeley/attention_detection_dataset_v3.csv"
df = pd.read_csv(csv_path)

print("=== Mendeley CSV Summary ===")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)

print("\n=== Missing Values ===")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Zero missing values across all columns!")

print("\n=== Duplicate Rows ===")
dup_count = df.duplicated().sum()
print(f"Exact duplicate rows: {dup_count} (out of {len(df)})")

print("\n=== Target Variable Distribution ===")
if "label" in df.columns:
    vc = df["label"].value_counts(dropna=False)
    print("Label Value Counts:\n", vc)
    print("Label Proportions:\n", df["label"].value_counts(normalize=True))
else:
    print("WARNING: 'label' column not found!")

print("\n=== Checking for Grouping/Session Metadata ===")
id_candidates = [c for c in df.columns if any(k in c.lower() for k in ["id", "session", "user", "student", "time", "frame", "date", "subject"])]
print("Potential Grouping Columns:", id_candidates)

print("\n=== Sample Head ===")
print(df.head(3))

print("\n=== Descriptive Statistics Summary ===")
print(df.describe().T[["count", "mean", "std", "min", "50%", "max"]].head(15))
