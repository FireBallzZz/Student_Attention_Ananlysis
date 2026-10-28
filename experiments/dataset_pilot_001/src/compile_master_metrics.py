import json
from pathlib import Path

out_dir = Path("experiments/dataset_pilot_001")

with open(out_dir / "metrics_tabular.json", "r") as f:
    metrics_a = json.load(f)

with open(out_dir / "metrics_scb_detector.json", "r") as f:
    metrics_b = json.load(f)

with open(out_dir / "metrics_ouc_audit.json", "r") as f:
    metrics_c = json.load(f)

# Downloaded files summary
downloaded_files = [
    {
        "filename": "attention_detection_dataset_v3.csv",
        "dataset": "Mendeley Students Attention Detection Dataset v3",
        "source_url": "https://data.mendeley.com/datasets/smzggbnkd2/3",
        "size_bytes": 1609237,
        "size_mb": 1.53,
        "md5": "1280c448eed54e264faf123e6add2e15",
        "sha256": "b577cb262d35f8b4b31db3bcb4132e82ed09aca0514f430178f3d3959154e685",
        "license": "CC BY 4.0"
    },
    {
        "filename": "scb_pilot (100 image-label pairs)",
        "dataset": "SCB-Dataset (SCB5 Handrise-Read-Write)",
        "source_url": "https://huggingface.co/datasets/wintonYF/SCB-Dataset",
        "size_bytes": 13421772,
        "size_mb": 12.80,
        "total_images": 100,
        "license": "Non-commercial Academic Research"
    }
]

master_metrics = {
    "experiment_id": "dataset_pilot_001",
    "timestamp": "2026-10-27T22:54:00Z",
    "hardware": {
        "gpu": "NVIDIA GeForce RTX 3060",
        "vram_gb": 12.0,
        "cpu_arch": "AMD64 (x86_64)",
        "os": "Windows 11"
    },
    "files_downloaded": downloaded_files,
    "experiments": {
        "experiment_a_tabular": {
            "category": "A. Individual attention classification from pre-extracted features",
            "result_type": "Proper validation result (Stratified Split)",
            "metrics": metrics_a
        },
        "experiment_b_scb": {
            "category": "B. Student action detection from images",
            "result_type": "Pipeline smoke test (Small Pilot)",
            "metrics": metrics_b
        },
        "experiment_c_ouc_cge": {
            "category": "C. Group-level engagement classification",
            "result_type": "Small pilot result (Multi-View Synchronized)",
            "metrics": metrics_c
        },
        "category_d_classroom_video_individual": {
            "category": "D. Individual attention classification from classroom video",
            "result_type": "BLOCKED / NO OFF-THE-SHELF BENCHMARK",
            "status": "No public dataset provides continuous multi-student classroom video with verified individual attention ground truth. Requires thesis annotation protocol execution."
        }
    }
}

with open(out_dir / "metrics.json", "w") as f:
    json.dump(master_metrics, f, indent=2)

print(f"Master metrics compiled to {out_dir / 'metrics.json'}")
