"""Phase Four: Pilot Data Validation Module.

Validates video integrity, timestamp alignment, duplicate frames, label distribution,
and creates diagnostic contact sheet visualizations.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import cv2
import numpy as np

from src.ingestion.video_reader import VideoReader, VideoMetadata


class PilotDataValidator:
    """Comprehensive validation suite for pilot classroom datasets."""

    def __init__(self, pilot_dir: Path, reports_dir: Path):
        self.pilot_dir = Path(pilot_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.reader = VideoReader(sample_fps=5.0)

    def find_video_clips(self) -> List[Path]:
        """Find all video files in pilot directory."""
        video_extensions = [".mp4", ".avi", ".mkv", ".mov"]
        clips = []
        for ext in video_extensions:
            clips.extend(self.pilot_dir.rglob(f"*{ext}"))
        return sorted(clips)

    def check_near_duplicates(self, video_path: Path, max_frames: int = 50, diff_threshold: float = 0.01) -> float:
        """
        Check for frozen / duplicate frame sequences.
        Returns the percentage of consecutive frames with near-zero pixel differences.
        """
        cap = cv2.VideoCapture(str(video_path))
        prev_gray = None
        duplicate_count = 0
        total_pairs = 0
        frame_idx = 0

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Sample every 6th frame (~5 fps on 30 fps video)
            if frame_idx % 6 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 90))  # downscale for fast perceptual diff
                if prev_gray is not None:
                    diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0
                    if diff < diff_threshold:
                        duplicate_count += 1
                    total_pairs += 1
                prev_gray = gray
            frame_idx += 1

        cap.release()
        return (duplicate_count / total_pairs) if total_pairs > 0 else 0.0

    def generate_contact_sheet(self, validated_clips: List[Dict[str, Any]], output_path: Path):
        """Create a diagnostic grid contact sheet from sampled pilot frames."""
        sample_thumbs = []
        thumb_w, thumb_h = 320, 180

        for clip_info in validated_clips[:8]:  # show up to 8 clips
            vpath = Path(clip_info["filepath"])
            cap = cv2.VideoCapture(str(vpath))
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Sample frame at 30% through the clip
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_f * 0.3))
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                thumb = cv2.resize(frame, (thumb_w, thumb_h))
                # Add label overlay banner
                label_text = f"{clip_info['label']} | {vpath.stem}"
                cv2.rectangle(thumb, (0, 0), (thumb_w, 24), (0, 0, 0), -1)
                cv2.putText(thumb, label_text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                sample_thumbs.append(thumb)

        if not sample_thumbs:
            return

        # Arrange into grid (up to 4 cols)
        cols = min(4, len(sample_thumbs))
        rows = int(np.ceil(len(sample_thumbs) / cols))
        grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

        for idx, thumb in enumerate(sample_thumbs):
            r = idx // cols
            c = idx % cols
            grid[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb

        cv2.imwrite(str(output_path), grid)
        print(f"[CONTACT SHEET] Generated diagnostic contact sheet: {output_path}")

    def run_validation(self) -> Dict[str, Any]:
        """Execute full Phase Four validation and produce report."""
        clips = self.find_video_clips()
        if not clips:
            raise FileNotFoundError(f"No video clips found in pilot directory: {self.pilot_dir}")

        validated_clips = []
        class_distribution = {}
        total_duration = 0.0
        all_passed = True

        for vpath in clips:
            meta = self.reader.inspect(vpath)
            label = vpath.parent.name
            class_distribution[label] = class_distribution.get(label, 0) + 1

            duplicate_ratio = self.check_near_duplicates(vpath)
            total_duration += meta.duration_sec

            is_frozen = duplicate_ratio > 0.95
            clip_result = {
                "filepath": str(vpath).replace("\\", "/"),
                "filename": vpath.name,
                "label": label,
                "width": meta.width,
                "height": meta.height,
                "fps": meta.fps,
                "total_frames": meta.total_frames,
                "duration_sec": round(meta.duration_sec, 2),
                "is_valid": meta.is_valid,
                "validation_errors": meta.validation_errors,
                "stillness_ratio": round(duplicate_ratio, 4),
                "is_frozen_video": is_frozen,
                "pass_status": meta.is_valid and (not is_frozen)
            }
            if not clip_result["pass_status"]:
                all_passed = False
            validated_clips.append(clip_result)

        # Contact sheet
        contact_sheet_path = self.reports_dir / "pilot_contact_sheet.png"
        self.generate_contact_sheet(validated_clips, contact_sheet_path)

        report = {
            "validation_timestamp": json.loads(json.dumps(Path(__file__).stat().st_mtime)),
            "pilot_directory": str(self.pilot_dir).replace("\\", "/"),
            "total_clips_checked": len(validated_clips),
            "class_distribution": class_distribution,
            "total_pilot_duration_sec": round(total_duration, 2),
            "all_clips_valid": all_passed,
            "label_level_warning": "OUC-CGE provides clip-level group engagement labels (High/Medium/Low), NOT individual student attention ground-truth. Detection, tracking, and pose extraction operate at individual student granularity.",
            "clips": validated_clips
        }

        report_file = self.reports_dir / "pilot_validation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("=" * 70)
        print("PILOT DATA VALIDATION SUMMARY")
        print(f"Total Clips Checked: {len(validated_clips)}")
        print(f"Class Distribution: {class_distribution}")
        print(f"Total Video Duration: {total_duration:.1f}s")
        print(f"Integrity Status: {'PASS' if all_passed else 'FAIL'}")
        print(f"Report written to: {report_file}")
        print("=" * 70)
        return report
