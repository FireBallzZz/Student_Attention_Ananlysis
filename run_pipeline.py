"""Command-Line Interface for Classroom Attention Video Pipeline.

Usage:
  python run_pipeline.py --input "data/clip 4.mp4" --output "outputs/clip_4_annotated.mp4"
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.video_pipeline import ClassroomAttentionPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Smart Classroom Attention Pipeline on any video")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input classroom video file")
    parser.add_argument("--output", "-o", type=str, default="web_app/static/outputs/output_annotated.mp4", help="Path to rendered output video")
    parser.add_argument("--fps", type=float, default=10.0, help="Sampling FPS for video processing (default: 10.0)")
    parser.add_argument("--drop_thresh", type=float, default=20.0, help="Attention drop threshold in percentage (default: 20.0)")
    parser.add_argument("--max_frames", "-m", type=int, default=None, help="Maximum frames to process (optional, default: process all)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING CLASSROOM ATTENTION PIPELINE")
    print(f"  Input Video : {input_path}")
    print(f"  Output Video: {output_path}")
    print(f"  Sample FPS  : {args.fps}")
    if args.max_frames:
        print(f"  Max Frames  : {args.max_frames}")
    print("=" * 80)

    pipeline = ClassroomAttentionPipeline(
        sample_fps=args.fps,
        drop_threshold=args.drop_thresh
    )

    def on_progress(p, msg):
        print(f"[{p*100:5.1f}%] {msg}")

    results = pipeline.process_video(
        input_video_path=input_path,
        output_video_path=output_path,
        max_frames=args.max_frames,
        progress_callback=on_progress
    )

    print("\n" + "=" * 80)
    print("CLASSROOM ATTENTION ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total Seated Census Count: {results['census_student_count']} students")
    print(f"Total Students Tracked   : {results['total_students_tracked']}")
    print(f"Exited Camera View       : {results['exited_camera_count']}")
    print(f"Class Average Attention  : {results['class_average_attention']}%")
    print(f"Attentive Percentage     : {results['attentive_percentage']}%")
    print(f"Inattentive Percentage   : {results['inattentive_percentage']}%")
    print(f"Attention Drop Events    : {results['drop_events_count']}")
    print(f"Annotated Video Saved    : {results['output_video_path']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
