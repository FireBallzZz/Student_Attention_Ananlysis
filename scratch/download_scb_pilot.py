import os
import sys
import zipfile
import hashlib
from pathlib import Path

sys.path.append(".")
from scratch.test_remote_zip import HttpRangeFile

def download_scb_pilot(n_train=70, n_val=30):
    out_dir = Path("experiments/dataset_pilot_001/data/scb_pilot")
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    url = "https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.zip"
    print("Opening remote zip via seekable HTTP range...")
    f = HttpRangeFile(url)
    zf = zipfile.ZipFile(f)
    namelist = set(zf.namelist())

    train_pairs = []
    val_pairs = []

    for name in sorted(namelist):
        if "images/train/" in name and name.endswith((".jpg", ".png")):
            lbl_name = name.replace("images/train/", "labels/train/").rsplit(".", 1)[0] + ".txt"
            if lbl_name in namelist:
                train_pairs.append((name, lbl_name))
        elif "images/val/" in name and name.endswith((".jpg", ".png")):
            lbl_name = name.replace("images/val/", "labels/val/").rsplit(".", 1)[0] + ".txt"
            if lbl_name in namelist:
                val_pairs.append((name, lbl_name))

    print(f"Total available pairs -> Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    selected_train = train_pairs[:n_train]
    selected_val = val_pairs[:n_val]

    download_log = []
    total_bytes = 0

    print(f"Extracting {len(selected_train)} train images & labels...")
    for idx, (img_name, lbl_name) in enumerate(selected_train):
        img_data = zf.read(img_name)
        lbl_data = zf.read(lbl_name)
        
        fname = Path(img_name).name
        lfname = Path(lbl_name).name

        (out_dir / "images" / "train" / fname).write_bytes(img_data)
        (out_dir / "labels" / "train" / lfname).write_bytes(lbl_data)

        total_bytes += len(img_data) + len(lbl_data)
        download_log.append({
            "split": "train",
            "image": fname,
            "image_bytes": len(img_data),
            "label_bytes": len(lbl_data),
            "image_sha256": hashlib.sha256(img_data).hexdigest()
        })
        if (idx + 1) % 10 == 0:
            print(f"  Train: {idx + 1}/{len(selected_train)} extracted")

    print(f"Extracting {len(selected_val)} val images & labels...")
    for idx, (img_name, lbl_name) in enumerate(selected_val):
        img_data = zf.read(img_name)
        lbl_data = zf.read(lbl_name)
        
        fname = Path(img_name).name
        lfname = Path(lbl_name).name

        (out_dir / "images" / "val" / fname).write_bytes(img_data)
        (out_dir / "labels" / "val" / lfname).write_bytes(lbl_data)

        total_bytes += len(img_data) + len(lbl_data)
        download_log.append({
            "split": "val",
            "image": fname,
            "image_bytes": len(img_data),
            "label_bytes": len(lbl_data),
            "image_sha256": hashlib.sha256(img_data).hexdigest()
        })
        if (idx + 1) % 10 == 0:
            print(f"  Val: {idx + 1}/{len(selected_val)} extracted")

    # Create dataset yaml for YOLO
    yaml_content = f"""path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

nc: 3
names: ['hand-raising', 'read', 'write']
"""
    (out_dir / "scb_pilot.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"\nDownloaded {len(download_log)} image-label pairs.")
    print(f"Total downloaded size: {total_bytes / (1024*1024):.2f} MB")
    print(f"Saved dataset YAML to: {out_dir / 'scb_pilot.yaml'}")

    import json
    (out_dir / "download_manifest.json").write_text(json.dumps({
        "dataset_name": "SCB-Dataset (SCB5 Handrise-Read-Write Pilot)",
        "source_archive": "SCB5-Handrise-Read-write-2024-9-17.zip",
        "total_images": len(download_log),
        "train_count": len(selected_train),
        "val_count": len(selected_val),
        "total_bytes": total_bytes,
        "files": download_log
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    download_scb_pilot(70, 30)
