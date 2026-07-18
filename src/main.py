"""Cortex-Vision main entry point — FastAPI application."""

import os
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Security
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .detector import Detector
from .pipeline import BatchPipeline
from .utils import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_VIDEO_SIZE,
    MAX_IMAGE_SIZE,
    ALLOWED_EXTENSIONS,
    process_video_frames,
    validate_image,
)

import time
import numpy as np
import cv2

app = FastAPI(title="Cortex-Vision", version="0.1.0")

# ---- Security: CORS ----
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---- Security: API Key Auth ----
API_KEY = os.getenv("API_KEY", "dev-key-do-not-use-in-production")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Dependency that validates API key for non-public endpoints."""
    if api_key is None or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ---- Security: Headers Middleware ----
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'"
        )
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# ---- Security: Rate Limiting ----
from collections import defaultdict
import time

RATE_LIMIT = int(os.getenv("RATE_LIMIT", "30"))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/"):
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - RATE_WINDOW
        self.requests[client_ip] = [t for t in self.requests[client_ip] if t > window_start]
        if len(self.requests[client_ip]) >= RATE_LIMIT:
            return JSONResponse(
                {"error": "Rate limit exceeded. Try again later."},
                status_code=429,
            )
        self.requests[client_ip].append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)

# ---- Helpers ----
def sanitize_filename(filename: str) -> str:
    """Remove path separators and dangerous characters from a filename."""
    # Keep only the name (strip any path components)
    safe = Path(filename).name
    # Remove any remaining dangerous characters
    safe = re.sub(r'[^\w.\- ]', '_', safe)
    # Limit length
    return safe[:128]

# Configuration
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Engine
detector = Detector(model_name=os.getenv("YOLO_MODEL", "yolov8n.pt"))
pipeline = BatchPipeline(detector)

# Store batch results in memory (in production: Redis/DB)
batch_results = {}

# Serve static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI (public, no auth required)."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return "<h1>Cortex-Vision API</h1><p>Static UI not found. Use /docs for API.</p>"


@app.post("/upload")
async def upload_image(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """Upload a single image and get detection results."""
    contents = await file.read()
    validate_image(file.filename, len(contents))

    safe_name = sanitize_filename(file.filename)
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    image_path.write_bytes(contents)

    results = detector.detect(str(image_path))

    return JSONResponse({
        "filename": file.filename,
        "detections": results,
    })


@app.post("/upload/batch")
async def upload_batch(files: list[UploadFile], api_key: str = Depends(verify_api_key)):
    """Upload multiple images for batch processing."""
    task_id = uuid.uuid4().hex
    file_paths = []

    for file in files:
        contents = await file.read()
        validate_image(file.filename, len(contents))

        safe_name = sanitize_filename(file.filename)
        image_path = UPLOAD_DIR / f"{task_id}_{safe_name}"
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
async def upload_video(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """Upload a video and get per-frame detection results.

    Extracts frames at 1 FPS and runs YOLO detection on each.
    """
    safe_name = sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return JSONResponse(
            {"error": f"Unsupported video format. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"},
            status_code=400,
        )

    video_id = uuid.uuid4().hex
    video_path = UPLOAD_DIR / f"{video_id}_{safe_name}"
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


@app.post("/detect/frame")
async def detect_frame(file: UploadFile = File(...)):
    """Receive a single frame from webcam and run detection.

    Lightweight endpoint — no disk I/O, processes entirely in memory.
    Called periodically (~1 FPS) by the live webcam UI.
    No auth required — only accepts small JPEG blobs, no file storage.
    """
    contents = await file.read()
    if not contents:
        return JSONResponse({"error": "Empty frame"}, status_code=400)

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image data"}, status_code=400)

    detections = detector.detect_array(img)

    return JSONResponse({
        "detections": detections,
        "object_count": len(detections),
        "timestamp": time.time(),
    })


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
    """Health check endpoint (public, no auth required)."""
    return {"status": "ok", "model": detector.model_name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
