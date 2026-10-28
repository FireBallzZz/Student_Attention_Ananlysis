import os
from pathlib import Path

def scan_dir(base_path):
    print(f"Scanning: {base_path}")
    base = Path(base_path)
    if not base.exists():
        print("  Does not exist.")
        return
    for item in sorted(base.rglob("*")):
        if item.is_file():
            rel = item.relative_to(base)
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  FILE: {rel} ({size_mb:.2f} MB)")

print("--- DATA DIRECTORY ---")
scan_dir("data")

print("\n--- EXPERIMENTS DIRECTORY ---")
scan_dir("experiments")

print("\n--- SEARCHING FOR CSV / JSON / ZIP / TXT DATASETS IN REPO ---")
for ext in ["*.csv", "*.zip", "*.parquet", "*.json"]:
    matches = list(Path(".").glob(f"**/{ext}"))
    matches = [m for m in matches if not ".git" in str(m) and not ".pytest_cache" in str(m)]
    print(f"Ext {ext}: {len(matches)} files")
    for m in matches[:10]:
        print(f"  {m} ({m.stat().st_size / 1024:.1f} KB)")
