"""Tests for Phase 2: Video Ingestion, Decoding, and Error Handling."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.ingestion.video_reader import VideoReader, VideoMetadata


@pytest.fixture
def temp_valid_video(tmp_path: Path) -> Path:
    """Create a temporary valid 15-frame MP4 video."""
    video_path = tmp_path / "valid_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for i in range(15):
        frame = np.full((240, 320, 3), fill_value=i * 15, dtype=np.uint8)
        out.write(frame)
    out.release()
    return video_path


def test_video_metadata_extraction(temp_valid_video: Path):
    """Verify that metadata extraction accurately identifies dimensions, FPS, and count."""
    reader = VideoReader(sample_fps=10.0)
    meta = reader.inspect(temp_valid_video)
    assert isinstance(meta, VideoMetadata)
    assert meta.is_valid is True
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == 10.0
    assert meta.total_frames == 15
    assert meta.duration_sec == pytest.approx(1.5, rel=1e-2)


def test_stream_frames_generator(temp_valid_video: Path):
    """Verify that stream_frames yields frames one-by-one without memory blowout."""
    reader = VideoReader(sample_fps=10.0)
    frames_yielded = 0
    for ts_frame in reader.stream_frames(temp_valid_video):
        assert ts_frame.frame.shape == (240, 320, 3)
        assert ts_frame.timestamp_sec >= 0.0
        frames_yielded += 1
    assert frames_yielded == 15


def test_nonexistent_file_inspection_and_stream():
    """Verify that missing files report invalid metadata and stream_frames raises ValueError."""
    reader = VideoReader()
    missing_path = Path("data/does_not_exist_xyz.mp4")
    meta = reader.inspect(missing_path)
    assert meta.is_valid is False
    assert "File does not exist" in meta.validation_errors[0]

    with pytest.raises(ValueError, match="Cannot stream corrupt video"):
        list(reader.stream_frames(missing_path))


def test_empty_file_error(tmp_path: Path):
    """Verify that 0-byte video files report invalid metadata and raise ValueError on stream."""
    empty_file = tmp_path / "empty_video.mp4"
    empty_file.touch()

    reader = VideoReader()
    meta = reader.inspect(empty_file)
    assert meta.is_valid is False
    assert "0 bytes" in meta.validation_errors[0]

    with pytest.raises(ValueError, match="0 bytes"):
        list(reader.stream_frames(empty_file))


def test_corrupt_file_error(tmp_path: Path):
    """Verify that corrupt non-video files report invalid metadata and raise ValueError on stream."""
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"THIS IS NOT A VALID VIDEO HEADER CONVERTED TO BYTES")

    reader = VideoReader()
    meta = reader.inspect(corrupt_file)
    assert meta.is_valid is False
    assert any("open" in err.lower() or "first frame" in err.lower() for err in meta.validation_errors)

    with pytest.raises(ValueError, match="Cannot stream corrupt video"):
        list(reader.stream_frames(corrupt_file))
