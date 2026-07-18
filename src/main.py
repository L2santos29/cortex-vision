"""Cortex-Vision main entry point — FastAPI application."""

import os
import uuid
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .detector import Detector
from .pipeline import BatchPipeline
from .utils import ALLOWED_VIDEO_EXTENSIONS, MAX_VIDEO_SIZE, process_video_frames

app = FastAPI(title="Cortex-Vision", version="0.1.0")

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Engine
detector = Detector()
pipeline = BatchPipeline(detector)

# Store batch results in memory (in production: Redis/DB)
batch_results = {}

# Serve static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return "<h1>Cortex-Vision API</h1><p>Static UI not found. Use /docs for API.</p>"


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload a single image and get detection results."""
    contents = await file.read()
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    image_path.write_bytes(contents)

    results = detector.detect(str(image_path))

    return JSONResponse({
        "filename": file.filename,
        "detections": results,
    })


@app.post("/upload/batch")
async def upload_batch(files: list[UploadFile]):
    """Upload multiple images for batch processing."""
    task_id = uuid.uuid4().hex
    file_paths = []

    for file in files:
        contents = await file.read()
        image_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
        image_path.write_bytes(contents)
        file_paths.append(str(image_path))

    # Process batch asynchronously
    results = pipeline.process(file_paths)
    aggregated = pipeline.aggregate(results)

    batch_results[task_id] = {
        "files_processed": len(files),
        "detections": aggregated,
        "per_file": results,
    }

    return JSONResponse({
        "task_id": task_id,
        "files_processed": len(files),
        "total_objects": len(aggregated),
    })


@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video and get per-frame detection results.

    Extracts frames at 1 FPS and runs YOLO detection on each.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return JSONResponse(
            {"error": f"Unsupported video format. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"},
            status_code=400,
        )

    video_id = uuid.uuid4().hex
    video_path = UPLOAD_DIR / f"{video_id}_{file.filename}"
    contents = await file.read()

    if len(contents) > MAX_VIDEO_SIZE:
        return JSONResponse({"error": "Video too large (>500MB)"}, status_code=400)

    video_path.write_bytes(contents)

    try:
        frames_data = process_video_frames(str(video_path), detector, fps_sample=1)
        duration = frames_data[-1]["timestamp"] if frames_data else 0

        return JSONResponse({
            "filename": file.filename,
            "video_id": video_id,
            "duration_seconds": duration,
            "total_frames_processed": len(frames_data),
            "total_objects": sum(f["object_count"] for f in frames_data),
            "frames": frames_data,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Clean up video file after processing
        if video_path.exists():
            video_path.unlink()


@app.get("/results/{task_id}")
async def get_results(task_id: str):
    """Get batch processing results."""
    if task_id not in batch_results:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return JSONResponse(batch_results[task_id])


@app.get("/export/{task_id}")
async def export_csv(task_id: str):
    """Export batch results as CSV."""
    if task_id not in batch_results:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    import csv
    import io

    detections = batch_results[task_id]["detections"]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["image", "class", "confidence", "bbox"])

    for d in detections:
        writer.writerow([
            d.get("image", ""),
            d.get("class", ""),
            d.get("confidence", 0),
            d.get("bbox", []),
        ])

    csv_path = OUTPUT_DIR / f"{task_id}.csv"
    csv_path.write_text(output.getvalue())

    return FileResponse(csv_path, filename=f"cortex-vision-{task_id}.csv")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model": detector.model_name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
