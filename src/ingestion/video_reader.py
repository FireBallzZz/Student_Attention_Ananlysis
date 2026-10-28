"""Stage 1 — Source and Extraction: Video Ingestion and Validation Module.

Provides robust video decoding, timestamp extraction, integrity checks,
and metadata preservation for classroom attention analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Metadata container for ingested video files."""
    filepath: str
    filename: str
    session_id: str
    clip_id: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    fourcc: str
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class TimestampedFrame:
    """Represents a decoded video frame with exact temporal context."""
    frame_idx: int
    timestamp_sec: float
    frame: np.ndarray
    clip_id: str
    session_id: str


class VideoReader:
    """Reads and validates classroom video files with configurable sampling."""

    def __init__(self, sample_fps: Optional[float] = None, target_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            sample_fps: Target frames per second to sample. If None, samples all native frames.
            target_size: Optional (width, height) tuple to resize frames to.
        """
        self.sample_fps = sample_fps
        self.target_size = target_size

    def inspect(self, video_path: Path) -> VideoMetadata:
        """Inspect video file integrity, stream properties, and metadata without decoding full video."""
        video_path = Path(video_path)
        errors = []

        if not video_path.exists():
            return VideoMetadata(
                filepath=str(video_path),
                filename=video_path.name,
                session_id="",
                clip_id="",
                width=0,
                height=0,
                fps=0.0,
                total_frames=0,
                duration_sec=0.0,
                fourcc="",
                is_valid=False,
                validation_errors=["File does not exist."]
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return VideoMetadata(
                filepath=str(video_path),
                filename=video_path.name,
                session_id=video_path.parent.name,
                clip_id=video_path.stem,
                width=0,
                height=0,
                fps=0.0,
                total_frames=0,
                duration_sec=0.0,
                fourcc="",
                is_valid=False,
                validation_errors=["OpenCV failed to open video stream."]
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        # Verify stream can read at least the first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            errors.append("First frame could not be read or is empty.")
        cap.release()

        if width <= 0 or height <= 0:
            errors.append(f"Invalid frame dimensions: {width}x{height}")
        if fps <= 0.0 or fps > 240.0:
            errors.append(f"Invalid FPS: {fps}")
        if total_frames <= 0:
            errors.append(f"Invalid total frame count: {total_frames}")

        duration_sec = total_frames / fps if fps > 0 else 0.0

        # Session and clip identification from path convention
        session_id = video_path.parent.name
        clip_id = video_path.stem

        return VideoMetadata(
            filepath=str(video_path),
            filename=video_path.name,
            session_id=session_id,
            clip_id=clip_id,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            fourcc=fourcc,
            is_valid=len(errors) == 0,
            validation_errors=errors
        )

    def stream_frames(self, video_path: Path) -> Generator[TimestampedFrame, None, None]:
        """
        Stream timestamped frames with configurable downsampling.
        Yields TimestampedFrame instances.
        """
        video_path = Path(video_path)
        metadata = self.inspect(video_path)
        if not metadata.is_valid:
            raise ValueError(f"Cannot stream corrupt video {video_path}: {metadata.validation_errors}")

        cap = cv2.VideoCapture(str(video_path))
        native_fps = metadata.fps
        sample_step = 1
        if self.sample_fps is not None and self.sample_fps > 0 and self.sample_fps < native_fps:
            sample_step = int(round(native_fps / self.sample_fps))

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % sample_step == 0:
                    timestamp_sec = frame_idx / native_fps

                    if self.target_size is not None:
                        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

                    yield TimestampedFrame(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        frame=frame,
                        clip_id=metadata.clip_id,
                        session_id=metadata.session_id
                    )

                frame_idx += 1
        finally:
            cap.release()
