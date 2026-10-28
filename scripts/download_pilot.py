"""Pilot Dataset Downloader for Classroom Attention Research.

Downloads a small, representative pilot subset of OUC-CGE from Open Science Framework (OSF).
Verifies cryptographic checksums (MD5, SHA256) and extracts clips to data/pilot/.
Generates data/manifests/pilot_manifest.json.
"""

import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
import requests

# Dataset Metadata
DATASET_METADATA = {
    "dataset_name": "OUC-CGE (Ocean University of China Classroom Group Engagement Dataset)",
    "official_doi": "10.17605/OSF.IO/BRD2C",
    "publication": "A Video Dataset for Classroom Group Engagement Recognition (Scientific Data, Nature, 2024)",
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "source_platform": "Open Science Framework (OSF)",
    "camera_specifications": "Fixed Logitech C922 Pro webcams, 1280x720, 30 FPS",
    "annotation_type": "Ordinal 3-level clip engagement (High, Medium, Low)",
    "pilot_purpose": "Stage 1-5 multi-student tracking, pose estimation, feature extraction, and attention drop benchmarking"
}

# Pilot archives to download
PILOT_ARCHIVES = [
    {
        "name": "low_sample.zip",
        "guid": "xcwj8",
        "url": "https://osf.io/download/xcwj8/",
        "size_bytes": 7954563,
        "md5": "523e367c6d7768faf58753236b88f4d8",
        "sha256": "4575e7fd9d433ffc1b6da1ba8a28eacd4feb95b7afdd0264cfd7a9f6599aa778",
        "label": "low_engagement",
        "engagement_level": 0
    },
    {
        "name": "high_sample.zip",
        "guid": "aw2kp",
        "url": "https://osf.io/download/aw2kp/",
        "size_bytes": 33896766,
        "md5": "c13340e4c86ea65eccdc2b3b1b7b542f",
        "sha256": "f6cae1ce079b5e698cdc9db3654d163da6a7b03877d7701c7f0d816c2281ee2d",
        "label": "high_engagement",
        "engagement_level": 2
    }
]


def compute_hashes(filepath: Path):
    """Compute MD5 and SHA256 hashes of a file."""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def download_file(url: str, dest_path: Path, expected_size: int = None):
    """Download file with streaming and basic progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        if expected_size and dest_path.stat().st_size == expected_size:
            print(f"[CACHE] {dest_path.name} already exists with correct size.")
            return True
        else:
            print(f"[RESUME] Partial or mismatched file found, re-downloading {dest_path.name}...")

    print(f"[DOWNLOADING] {dest_path.name} from {url}...")
    headers = {"User-Agent": "ClassroomAttentionResearchPilot/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
    print(f"[COMPLETE] Downloaded {downloaded:,} bytes to {dest_path.name}")
    return True


def extract_zip(zip_path: Path, extract_dir: Path):
    """Extract zip archive safely into directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Avoid path traversal
            if member.startswith("/") or ".." in member:
                continue
            zf.extract(member, extract_dir)
            extracted_path = extract_dir / member
            if extracted_path.is_file():
                extracted_files.append(extracted_path)
    return extracted_files


def run_pilot_download(project_root: Path):
    """Execute download, validation, extraction, and manifest generation."""
    raw_dir = project_root / "data" / "raw"
    pilot_dir = project_root / "data" / "pilot"
    manifests_dir = project_root / "data" / "manifests"

    raw_dir.mkdir(parents=True, exist_ok=True)
    pilot_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    total_download_bytes = 0

    print("=" * 70)
    print("PHASE THREE: PILOT DATASET DOWNLOAD (OUC-CGE)")
    print(f"Target Root: {project_root}")
    print("=" * 70)

    for item in PILOT_ARCHIVES:
        archive_path = raw_dir / item["name"]
        download_file(item["url"], archive_path, item["size_bytes"])

        # Checksum Verification
        md5_actual, sha256_actual = compute_hashes(archive_path)
        print(f"Verifying {item['name']}:")
        print(f"  MD5 Expected: {item['md5']}")
        print(f"  MD5 Actual:   {md5_actual}")
        assert md5_actual == item["md5"], f"MD5 mismatch for {item['name']}"
        assert sha256_actual == item["sha256"], f"SHA256 mismatch for {item['name']}"
        print(f"  Status: CHECKSUM PASSED (MD5 & SHA-256 match exactly)")

        total_download_bytes += archive_path.stat().st_size

        # Extraction into data/pilot/<label>/
        target_subfolder = pilot_dir / item["label"]
        extracted = extract_zip(archive_path, target_subfolder)
        print(f"  Extracted {len(extracted)} files to {target_subfolder.relative_to(project_root)}")

        for fpath in extracted:
            if fpath.suffix.lower() in [".mp4", ".avi", ".mkv"]:
                f_md5, f_sha256 = compute_hashes(fpath)
                manifest_entries.append({
                    "relative_path": str(fpath.relative_to(project_root)).replace("\\", "/"),
                    "filename": fpath.name,
                    "archive_source": item["name"],
                    "group_engagement_label": item["label"],
                    "engagement_ordinal": item["engagement_level"],
                    "file_size_bytes": fpath.stat().st_size,
                    "md5": f_md5,
                    "sha256": f_sha256
                })

    # Save manifest
    manifest_doc = {
        "metadata": DATASET_METADATA,
        "download_timestamp": datetime.utcnow().isoformat() + "Z",
        "total_archives": len(PILOT_ARCHIVES),
        "total_download_size_bytes": total_download_bytes,
        "total_video_clips": len(manifest_entries),
        "clips": manifest_entries
    }

    manifest_file = manifests_dir / "pilot_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)

    print("=" * 70)
    print(f"PILOT DOWNLOAD COMPLETE")
    print(f"Total Video Clips: {len(manifest_entries)}")
    print(f"Total Download Size: {total_download_bytes / (1024 * 1024):.2f} MB")
    print(f"Manifest written to: {manifest_file.relative_to(project_root)}")
    print("=" * 70)
    return manifest_doc


if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    run_pilot_download(project_root)
