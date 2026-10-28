"""FastAPI Web Server for Classroom Attention Analysis.

Provides:
- Static Webpage hosting with rich interactive UI
- Video upload & sample video selection
- End-to-end execution of ClassroomAttentionPipeline
- HTTP Range streaming for browser video playback
- Analysis results API & JSON export
"""

import os
import sys
import time
import json
import uuid
import mimetypes
from pathlib import Path
from typing import Optional, Dict

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.video_pipeline import ClassroomAttentionPipeline

app = FastAPI(
    title="Smart Classroom Attention Analytics",
    description="CSE Thesis Research Pipeline & Interactive Web Application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = PROJECT_ROOT / "web_app" / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
OUTPUTS_DIR = STATIC_DIR / "outputs"
PILOT_DIR = PROJECT_ROOT / "data" / "pilot"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory store for active analyses
active_jobs: Dict[str, Dict] = {}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/sample_videos")
async def get_sample_videos():
    """Return available verified classroom clips for quick testing."""
    samples = [
        {
            "id": "high_view1",
            "name": "Front-Center Perspective (High Engagement)",
            "filename": "view1.mp4",
            "rel_path": "high_engagement/high/view1.mp4",
            "description": "Front-row focus, high illumination, clear eye gaze & facial orientation.",
            "students_expected": "6–8 visible students",
            "camera_angle": "Frontal (View 1)",
            "duration_sec": 11.5
        },
        {
            "id": "low_view3",
            "name": "Side-Angle Oblique (Low Engagement)",
            "filename": "view3.mp4",
            "rel_path": "low_engagement/low/view3.mp4",
            "description": "Side-wall 45-degree angle, desk occlusion, posture slump detection.",
            "students_expected": "8–12 visible students",
            "camera_angle": "Side Oblique (View 3)",
            "duration_sec": 13.5
        },
        {
            "id": "high_view11",
            "name": "Rear High-Angle (Distance Gating Benchmark)",
            "filename": "view11.mp4",
            "rel_path": "high_engagement/high/view11.mp4",
            "description": "Elevated rear perspective, distant rows (<24px faces), heavy occlusion gating.",
            "students_expected": "10–15 visible students",
            "camera_angle": "Rear High (View 11)",
            "duration_sec": 11.5
        },
        {
            "id": "clip_4",
            "name": "Clip 4 (Ground-Truth Annotated Benchmark)",
            "filename": "clip 4.mp4",
            "rel_path": "clip 4.mp4",
            "description": "21 annotated students across 300 frames with ground-truth High/Moderate/Low labels.",
            "students_expected": "21 visible students (100% Discovered & Tracked)",
            "camera_angle": "Full Classroom Overview",
            "duration_sec": 10.0
        }
    ]
    return JSONResponse(content={"samples": samples})


def range_streaming_response(request: Request, file_path: Path):
    """Serve video with HTTP 206 Partial Content for browser seeking."""
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    mime_type, _ = mimetypes.guess_type(str(file_path))
    mime_type = mime_type or "video/mp4"

    if range_header:
        # e.g., "bytes=0-1024"
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0])
        end = int(byte_range[1]) if byte_range[1] else file_size - 1
        length = end - start + 1

        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                bytes_left = length
                chunk_size = 1024 * 1024
                while bytes_left > 0:
                    read_bytes = min(chunk_size, bytes_left)
                    data = f.read(read_bytes)
                    if not data:
                        break
                    bytes_left -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": mime_type,
        }
        return StreamingResponse(iter_file(), status_code=status.HTTP_206_PARTIAL_CONTENT, headers=headers)
    else:
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": mime_type,
        }
        return FileResponse(file_path, headers=headers)


@app.get("/api/video/{category}/{filename}")
async def stream_video(category: str, filename: str, request: Request):
    """Stream input or output video file."""
    if category == "sample":
        # Search in pilot dir or data root
        target = None
        for p in list(PILOT_DIR.rglob(filename)) + list((PROJECT_ROOT / "data").glob(filename)):
            target = p
            break
        if not target or not target.exists():
            raise HTTPException(status_code=404, detail=f"Sample video {filename} not found")
        return range_streaming_response(request, target)
    elif category == "upload":
        target = UPLOADS_DIR / filename
        return range_streaming_response(request, target)
    elif category == "output":
        target = OUTPUTS_DIR / filename
        return range_streaming_response(request, target)
    else:
        raise HTTPException(status_code=400, detail="Invalid video category")


@app.post("/api/analyze")
async def analyze_video(
    sample_id: Optional[str] = Form(None),
    video_file: Optional[UploadFile] = File(None),
    max_duration_sec: Optional[float] = Form(10.0),
    sample_fps: Optional[float] = Form(10.0),
    drop_threshold: Optional[float] = Form(20.0)
):
    """
    Execute full classroom attention pipeline on uploaded video or selected sample.
    """
    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    
    # 1. Resolve Input File
    if video_file and video_file.filename:
        safe_name = f"{job_id}_{video_file.filename}"
        input_path = UPLOADS_DIR / safe_name
        contents = await video_file.read()
        input_path.write_bytes(contents)
        input_video_type = "upload"
        input_video_url = f"/api/video/upload/{safe_name}"
    elif sample_id:
        sample_map = {
            "high_view1": PILOT_DIR / "high_engagement" / "high" / "view1.mp4",
            "low_view3": PILOT_DIR / "low_engagement" / "low" / "view3.mp4",
            "high_view11": PILOT_DIR / "high_engagement" / "high" / "view11.mp4",
            "clip_4": PROJECT_ROOT / "data" / "clip 4.mp4",
        }
        if sample_id not in sample_map:
            raise HTTPException(status_code=400, detail=f"Unknown sample ID: {sample_id}")
        input_path = sample_map[sample_id]
        input_video_type = "sample"
        input_video_url = f"/api/video/sample/{input_path.name}"
    else:
        raise HTTPException(status_code=400, detail="Must provide either video_file or sample_id")

    # 2. Output Video Path
    output_filename = f"output_{job_id}.mp4"
    output_path = OUTPUTS_DIR / output_filename
    output_video_url = f"/api/video/output/{output_filename}"

    print(f"\n[Server] Starting Pipeline Execution for Job {job_id}...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")

    # 3. Run Pipeline
    pipeline = ClassroomAttentionPipeline(
        sample_fps=sample_fps or 10.0,
        drop_threshold=drop_threshold or 20.0
    )

    max_frames = int((max_duration_sec or 10.0) * (sample_fps or 10.0))

    try:
        results = pipeline.process_video(
            input_video_path=input_path,
            output_video_path=output_path,
            max_frames=max_frames
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

    # 4. Attach URLs and metadata
    results["job_id"] = job_id
    results["input_video_url"] = input_video_url
    results["output_video_url"] = output_video_url
    results["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save analysis JSON
    analysis_json_file = OUTPUTS_DIR / f"analysis_{job_id}.json"
    with open(analysis_json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    active_jobs[job_id] = results
    return JSONResponse(content=results)


@app.get("/api/analysis/{job_id}")
async def get_analysis(job_id: str):
    if job_id in active_jobs:
        return JSONResponse(content=active_jobs[job_id])
    
    file_path = OUTPUTS_DIR / f"analysis_{job_id}.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return JSONResponse(content=data)
    raise HTTPException(status_code=404, detail="Analysis job not found")


if __name__ == "__main__":
    print("=" * 80)
    print("STARTING CLASSROOM ATTENTION RESEARCH SERVER")
    print("Web UI available at: http://127.0.0.1:8000")
    print("=" * 80)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
