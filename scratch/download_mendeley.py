import os
import requests
import hashlib
from pathlib import Path

out_dir = Path("experiments/dataset_pilot_001/data/mendeley")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "attention_detection_dataset_v3.csv"

download_url = "https://data.mendeley.com/public-files/datasets/smzggbnkd2/files/290fc681-61c7-4666-ad92-f2544b10790f/file_downloaded"
print(f"Downloading Mendeley dataset v3 from: {download_url}")

r = requests.get(download_url, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=30)
print("Status:", r.status_code)
if r.status_code == 200:
    content = r.content
    out_file.write_bytes(content)
    size_bytes = len(content)
    md5 = hashlib.md5(content).hexdigest()
    sha256 = hashlib.sha256(content).hexdigest()
    print(f"Saved: {out_file}")
    print(f"Size: {size_bytes} bytes ({size_bytes / (1024*1024):.2f} MB)")
    print(f"MD5: {md5}")
    print(f"SHA-256: {sha256}")
else:
    print("Download failed!")
