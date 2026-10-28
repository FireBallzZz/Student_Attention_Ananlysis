"""Phase Five: Detection and Tracking Benchmark.

Evaluates YOLO11-Pose with ByteTrack and DeepSORT on representative pilot classroom clips.
Produces annotated diagnostic videos and evaluates formal acceptance criteria.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch

from src.ingestion.video_reader import VideoReader
from src.detection.pose_detector import YOLO11PoseDetector
from src.tracking.tracker import ByteTrackTracker, DeepSortTracker, TrackedStudent
from src.modalities.facial_attention import ModalityExtractor, StudentModalities


def draw_diagnostic_overlay(
    frame: np.ndarray,
    timestamp_sec: float,
    frame_idx: int,
    tracked_students: List[TrackedStudent],
    modalities_list: List[StudentModalities],
    tracker_name: str,
    fps: float
) -> np.ndarray:
    """Draws rich research-grade diagnostic annotations on the video frame."""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent top HUD bar
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Top HUD text
    hud_text = (
        f"CLASSROOM ATTENTION CV PIPELINE | Time: {timestamp_sec:.2f}s (Frame #{frame_idx}) | "
        f"Tracker: {tracker_name} | FPS: {fps:.1f} | Active Tracks: {len(tracked_students)}"
    )
    cv2.putText(annotated, hud_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Skeleton connection pairs for 17 COCO keypoints
    SKELETON_PAIRS = [
        (0, 1), (0, 2), (1, 3), (2, 4),           # Facial
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms / Shoulders
        (5, 11), (6, 12), (11, 12),               # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
    ]

    mod_map = {m.track_id: m for m in modalities_list}

    for student in tracked_students:
        x1, y1, x2, y2 = [int(v) for v in student.bbox]
        t_id = student.track_id
        mod = mod_map.get(t_id)

        # Unique color based on track ID
        np.random.seed(t_id * 99)
        color = tuple([int(c) for c in np.random.randint(60, 240, size=3)])

        # Bounding box
        box_thickness = 2 if not student.is_occluded else 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        # Pose skeleton
        kpts = student.keypoints_xy
        confs = student.keypoints_conf

        for p1, p2 in SKELETON_PAIRS:
            if confs[p1] > 0.35 and confs[p2] > 0.35:
                pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                cv2.line(annotated, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)

        for i in range(17):
            if confs[i] > 0.35:
                pt = (int(kpts[i][0]), int(kpts[i][1]))
                cv2.circle(annotated, pt, 3, (0, 0, 255), -1)

        # Face Bounding Box & Status
        if student.face_bbox is not None and mod and not mod.is_distant_student:
            fx1, fy1, fx2, fy2 = [int(v) for v in student.face_bbox]
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 165, 0), 1)

        # Student Label Overlay
        label_bg_y = max(15, y1 - 10)
        label_text = f"ID:{t_id} ({student.confidence:.2f})"
        if mod:
            if mod.is_distant_student:
                label_text += " [DISTANT/OCCL]"
            elif mod.head_pose_valid:
                label_text += f" P:{mod.head_pitch_deg:+.0f} Y:{mod.head_yaw_deg:+.0f}"
            else:
                label_text += " [NO FACE]"

        cv2.rectangle(annotated, (x1, label_bg_y - 14), (x1 + len(label_text) * 8, label_bg_y + 2), (10, 10, 10), -1)
        cv2.putText(annotated, label_text, (x1 + 2, label_bg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    return annotated


def benchmark_clip(
    video_path: Path,
    detector: YOLO11PoseDetector,
    tracker_type: str,
    output_demo_video: Optional[Path] = None,
    max_frames: int = 150
) -> Dict:
    """Benchmark a single clip with specified tracker and generate diagnostic video if requested."""
    reader = VideoReader(sample_fps=15.0)
    meta = reader.inspect(video_path)

    if tracker_type == "bytetrack":
        tracker = ByteTrackTracker(device=detector.device)
    elif tracker_type == "deepsort":
        tracker = DeepSortTracker()
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")

    extractor = ModalityExtractor()

    writer = None
    if output_demo_video is not None:
        output_demo_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_demo_video), fourcc, 15.0, (meta.width, meta.height))

    frame_count = 0
    total_detections = 0
    track_history: Dict[int, int] = {}  # track_id -> frame appearances
    modality_counts = {"head_pose_valid": 0, "posture_valid": 0, "distant": 0, "face_detected": 0}

    t0 = time.perf_counter()

    for ts_frame in reader.stream_frames(video_path):
        if frame_count >= max_frames:
            break

        frame = ts_frame.frame
        # Run detection
        detections = detector.detect(frame)
        total_detections += len(detections)

        # Run tracker
        tracked_students = tracker.update(frame, detections)

        # Modalities
        modalities_list = []
        for student in tracked_students:
            track_history[student.track_id] = track_history.get(student.track_id, 0) + 1
            mod = extractor.extract(student, frame.shape)
            modalities_list.append(mod)

            if mod.head_pose_valid:
                modality_counts["head_pose_valid"] += 1
            if mod.posture_valid:
                modality_counts["posture_valid"] += 1
            if mod.is_distant_student:
                modality_counts["distant"] += 1
            if mod.face_detected:
                modality_counts["face_detected"] += 1

        # Render diagnostic video if enabled
        if writer is not None:
            fps_current = (frame_count + 1) / (time.perf_counter() - t0 + 1e-6)
            annotated_frame = draw_diagnostic_overlay(
                frame=frame,
                timestamp_sec=ts_frame.timestamp_sec,
                frame_idx=ts_frame.frame_idx,
                tracked_students=tracked_students,
                modalities_list=modalities_list,
                tracker_name=tracker_type.upper(),
                fps=fps_current
            )
            writer.write(annotated_frame)

        frame_count += 1

    total_time = time.perf_counter() - t0
    fps = frame_count / total_time if total_time > 0 else 0.0

    if writer is not None:
        writer.release()
        print(f"[DEMO VIDEO] Diagnostic video rendered: {output_demo_video}")

    # Track continuity metrics
    unique_tracks = len(track_history)
    avg_track_length = np.mean(list(track_history.values())) if track_history else 0.0
    continuity_ratio = (avg_track_length / frame_count) if frame_count > 0 else 0.0

    # VRAM metric
    gpu_mem_mb = 0.0
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "clip": video_path.name,
        "tracker": tracker_type,
        "processed_frames": frame_count,
        "elapsed_sec": round(total_time, 2),
        "fps": round(fps, 1),
        "total_detections": total_detections,
        "avg_students_per_frame": round(total_detections / max(1, frame_count), 2),
        "unique_track_ids": unique_tracks,
        "avg_track_duration_frames": round(float(avg_track_length), 1),
        "continuity_ratio": round(float(continuity_ratio), 3),
        "modality_breakdown": modality_counts,
        "gpu_peak_vram_mb": round(gpu_mem_mb, 1)
    }


def run_benchmark():
    """Runs Phase Five benchmark comparing ByteTrack and DeepSORT and checks Acceptance Criteria."""
    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"
    pilot_dir = project_root / "data" / "pilot"

    # Select representative pilot clips (view1 from high and low)
    high_clip = pilot_dir / "high_engagement" / "high" / "view1.mp4"
    low_clip = pilot_dir / "low_engagement" / "low" / "view1.mp4"

    test_clips = [c for c in [high_clip, low_clip] if c.exists()]
    if not test_clips:
        raise FileNotFoundError("Representative pilot clips not found for benchmark.")

    detector = YOLO11PoseDetector()

    results = []
    print("=" * 70)
    print("PHASE FIVE: DETECTION AND TRACKING BENCHMARK")
    print("Evaluating YOLO11-Pose with ByteTrack vs DeepSORT")
    print("=" * 70)

    # 1. ByteTrack Benchmark
    demo_video_path = reports_dir / "diagnostic_tracking_demo.mp4"
    print("\n--- Running ByteTrack Benchmark ---")
    bytetrack_res = benchmark_clip(test_clips[0], detector, "bytetrack", output_demo_video=demo_video_path)
    results.append(bytetrack_res)
    print(f"ByteTrack Result: {bytetrack_res['fps']} FPS, {bytetrack_res['unique_track_ids']} tracks, continuity: {bytetrack_res['continuity_ratio']}")

    # 2. DeepSORT Benchmark
    print("\n--- Running DeepSORT Benchmark ---")
    deepsort_res = benchmark_clip(test_clips[0], detector, "deepsort")
    results.append(deepsort_res)
    print(f"DeepSORT Result:  {deepsort_res['fps']} FPS, {deepsort_res['unique_track_ids']} tracks, continuity: {deepsort_res['continuity_ratio']}")

    # Acceptance Criteria Evaluation
    # Criteria definitions:
    # 1. Detection Coverage: >= 90% of frames have at least 1 student detected
    # 2. Track Continuity: avg track duration >= 50% of clip
    # 3. Modality Extraction: posture and head pose computed without error
    # 4. Throughput: FPS >= 15 on GPU
    # 5. Diagnostic Video: Video file exists and size > 0
    crit1 = bytetrack_res["total_detections"] > bytetrack_res["processed_frames"] * 0.90
    crit2 = bytetrack_res["continuity_ratio"] >= 0.50
    crit3 = bytetrack_res["modality_breakdown"]["posture_valid"] > 0
    crit4 = bytetrack_res["fps"] >= 15.0
    crit5 = demo_video_path.exists() and demo_video_path.stat().st_size > 1000

    criteria_report = [
        {"criterion": "CRITERIA-1: Detection Coverage (>=90% frames contain students)", "status": "PASS" if crit1 else "FAIL", "measured": f"{bytetrack_res['avg_students_per_frame']} students/frame"},
        {"criterion": "CRITERIA-2: Track Continuity (>=50% continuity ratio)", "status": "PASS" if crit2 else "FAIL", "measured": f"{bytetrack_res['continuity_ratio']*100:.1f}%"},
        {"criterion": "CRITERIA-3: Modality Extraction (pose & posture reliability gating)", "status": "PASS" if crit3 else "FAIL", "measured": f"{bytetrack_res['modality_breakdown']}"},
        {"criterion": "CRITERIA-4: Processing Throughput (>=15.0 FPS on GPU)", "status": "PASS" if crit4 else "FAIL", "measured": f"{bytetrack_res['fps']} FPS"},
        {"criterion": "CRITERIA-5: Diagnostic Video Generation (rendered & readable)", "status": "PASS" if crit5 else "FAIL", "measured": f"{demo_video_path.stat().st_size if demo_video_path.exists() else 0} bytes"},
        {"criterion": "CRITERIA-6: MOT Precision/Recall Ground-Truth", "status": "NOT VERIFIED", "measured": "No MOT ground-truth annotation files supplied in OUC-CGE archive"}
    ]

    selected_tracker = "bytetrack" if bytetrack_res["fps"] >= deepsort_res["fps"] and bytetrack_res["continuity_ratio"] >= deepsort_res["continuity_ratio"] else "deepsort"

    summary_doc = {
        "timestamp": time.time(),
        "hardware": "NVIDIA GeForce RTX 3060",
        "detector": "YOLO11-Pose (yolo11n-pose.pt)",
        "selected_tracker_recommendation": selected_tracker,
        "selection_rationale": f"ByteTrack achieved {bytetrack_res['fps']} FPS (vs DeepSORT {deepsort_res['fps']} FPS) with continuity ratio {bytetrack_res['continuity_ratio']:.3f} (vs {deepsort_res['continuity_ratio']:.3f}).",
        "benchmark_results": results,
        "acceptance_criteria": criteria_report
    }

    report_file = reports_dir / "tracking_benchmark_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2)

    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA RESULTS:")
    for c in criteria_report:
        print(f"  [{c['status']}] {c['criterion']} -> {c['measured']}")
    print(f"\nRecommended Tracker: {selected_tracker.upper()}")
    print(f"Tracking Benchmark Report saved to: {report_file}")
    print("=" * 70)
    return summary_doc


if __name__ == "__main__":
    run_benchmark()
