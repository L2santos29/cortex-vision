"""Cortex-Vision main entry point — FastAPI application."""

import asyncio
import csv
import io
import logging
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import ValidationError
from pydantic_settings import BaseSettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .detector import Detector
from .pipeline import BatchPipeline
from .utils import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_VIDEO_SIZE,
    process_video_frames,
    validate_image,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (ARC-02)
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    api_key: str  # Required — raises RuntimeError if API_KEY env var is missing
    cors_origins: str = "http://localhost:8000"
    upload_dir: str = "uploads"
    output_dir: str = "output"
    yolo_model: str = "yolov8n.pt"
    rate_limit: int = 30
    rate_window: int = 60
    max_batch_results: int = 1000

    model_config = {"env_prefix": ""}


try:
    settings = Settings()
except ValidationError as exc:
    raise RuntimeError(
        "API_KEY environment variable is required — set it or create a .env file."
    ) from exc

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Cortex-Vision", version="0.1.0")

# Compression (PER-07)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS (SEC-04)
ALLOWED_ORIGINS = settings.cors_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type", "Accept"],
)

# ---- Security: API Key Auth (SEC-02/SEC-03) ----
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Dependency that validates API key for non-public endpoints."""
    if api_key is None or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ---- Middleware: Security Headers (SEC-06) ----
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
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

# ---- Middleware: Rate Limiting ----
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/"):
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - settings.rate_window
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > window_start
        ]
        if len(self.requests[client_ip]) >= settings.rate_limit:
            return JSONResponse(
                {"error": "Rate limit exceeded. Try again later."},
                status_code=429,
            )
        self.requests[client_ip].append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)

# ---- Middleware: Request ID (MON-01/MON-02/MON-05) ----
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        logger.info(
            "Request started: %s %s [request_id=%s]",
            request.method,
            request.url.path,
            request_id,
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)

# ---- Directories ----
UPLOAD_DIR = Path(settings.upload_dir)
OUTPUT_DIR = Path(settings.output_dir)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- Engine ----
detector = Detector(model_name=settings.yolo_model)
pipeline = BatchPipeline(detector)

# In-memory batch result store (production: Redis/DB)
batch_results: dict[str, dict] = {}

# Serve static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---- Helpers ----
def sanitize_filename(filename: str) -> str:
    """Remove path separators and dangerous characters from a filename."""
    safe = Path(filename).name
    safe = re.sub(r"[^\w.\- ]", "_", safe)
    return safe[:128]


def _evict_oldest_batch() -> None:
    """Remove oldest entry when batch_results exceeds the limit (ARC-08)."""
    if len(batch_results) >= settings.max_batch_results:
        oldest = min(batch_results.keys(), key=lambda k: batch_results[k].get("_ts", 0))
        del batch_results[oldest]


# ---------------------------------------------------------------------------
# Routes — public (no auth) at root level (ARC-09)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve the web UI (public, no auth required)."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return "<h1>Cortex-Vision API</h1><p>Static UI not found. Use /docs for API.</p>"


@app.get("/health")
async def health() -> dict:
    """Health check endpoint (public, no auth required)."""
    return {"status": "ok", "model": detector.model_name}


# ---------------------------------------------------------------------------
# Routes — v1 API (auth required)
# ---------------------------------------------------------------------------

@app.post("/v1/upload", status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
) -> JSONResponse:
    """Upload a single image and get detection results."""
    contents = await file.read()
    validate_image(file.filename or "unknown", len(contents))

    safe_name = sanitize_filename(file.filename or "unknown")
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"

    try:
        image_path.write_bytes(contents)
    except OSError as exc:
        logger.error("Failed to write uploaded image %s: %s", image_path, exc)
        return JSONResponse({"error": "Failed to save uploaded file"}, status_code=500)

    try:
        results = await asyncio.to_thread(detector.detect, str(image_path))
    except Exception as exc:
        logger.error("Detection failed for %s: %s", image_path, exc)
        return JSONResponse({"error": "Detection failed"}, status_code=500)

    logger.info("Processed image %s (%d detections)", file.filename, len(results))
    return JSONResponse({"filename": file.filename, "detections": results}, status_code=201)


@app.post("/v1/upload/batch", status_code=201)
async def upload_batch(
    files: list[UploadFile],
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
) -> JSONResponse:
    """Upload multiple images for batch processing (background task)."""
    task_id = uuid.uuid4().hex
    file_paths: list[str] = []

    for file in files:
        contents = await file.read()
        validate_image(file.filename or "unknown", len(contents))

        safe_name = sanitize_filename(file.filename or "unknown")
        image_path = UPLOAD_DIR / f"{task_id}_{safe_name}"

        try:
            image_path.write_bytes(contents)
        except OSError as exc:
            logger.error("Failed to write batch file %s: %s", image_path, exc)
            continue

        file_paths.append(str(image_path))

    # Schedule background processing (PER-02)
    background_tasks.add_task(_process_batch, task_id, file_paths)

    return JSONResponse(
        {
            "task_id": task_id,
            "files_processed": len(file_paths),
        },
        status_code=201,
    )


async def _process_batch(task_id: str, file_paths: list[str]) -> None:
    """Background task: run detection and store results."""
    logger.info("Batch %s: processing %d files", task_id, len(file_paths))
    try:
        results = await asyncio.to_thread(pipeline.process, file_paths)
        aggregated = pipeline.aggregate(results)
        computed_stats = pipeline.stats(aggregated)

        _evict_oldest_batch()

        batch_results[task_id] = {
            "files_processed": len(file_paths),
            "detections": aggregated,
            "per_file": results,
            "stats": computed_stats,
            "_ts": time.time(),
        }
        logger.info("Batch %s: completed (%d objects)", task_id, len(aggregated))
    except Exception as exc:
        logger.error("Batch %s failed: %s", task_id, exc)


@app.post("/v1/upload/video", status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
) -> JSONResponse:
    """Upload a video and get per-frame detection results."""
    safe_name = sanitize_filename(file.filename or "unknown")
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return JSONResponse(
            {
                "error": (
                    f"Unsupported video format. Allowed: "
                    f"{', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
                )
            },
            status_code=400,
        )

    video_id = uuid.uuid4().hex
    video_path = UPLOAD_DIR / f"{video_id}_{safe_name}"
    contents = await file.read()

    if len(contents) > MAX_VIDEO_SIZE:
        return JSONResponse({"error": "Video too large (>500MB)"}, status_code=400)

    try:
        video_path.write_bytes(contents)
    except OSError as exc:
        logger.error("Failed to write video %s: %s", video_path, exc)
        return JSONResponse({"error": "Failed to save uploaded video"}, status_code=500)

    try:
        frames_data = process_video_frames(str(video_path), detector, fps_sample=1)
        duration = frames_data[-1]["timestamp"] if frames_data else 0

        logger.info(
            "Processed video %s (%d frames)",
            file.filename,
            len(frames_data),
        )
        return JSONResponse(
            {
                "filename": file.filename,
                "video_id": video_id,
                "duration_seconds": duration,
                "total_frames_processed": len(frames_data),
                "total_objects": sum(f["object_count"] for f in frames_data),
                "frames": frames_data,
            },
            status_code=201,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Video processing failed for %s: %s", video_path, exc)
        return JSONResponse({"error": "Video processing failed"}, status_code=500)
    except Exception as exc:
        logger.error("Unexpected error processing video %s: %s", video_path, exc)
        return JSONResponse({"error": "Video processing failed"}, status_code=500)
    finally:
        if video_path.exists():
            video_path.unlink()


@app.post("/v1/detect/frame")
async def detect_frame(
    file: UploadFile = File(...),
) -> JSONResponse:
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

    detections = await asyncio.to_thread(detector.detect_array, img)

    return JSONResponse({
        "detections": detections,
        "object_count": len(detections),
        "timestamp": time.time(),
    })


@app.get("/v1/results/{task_id}")
async def get_results(
    task_id: str,
    limit: int = 100,
    offset: int = 0,
    api_key: str = Depends(verify_api_key),
) -> JSONResponse:
    """Get batch processing results with pagination (PER-03)."""
    if task_id not in batch_results:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    result = dict(batch_results[task_id])
    detections = result.get("detections", [])
    result["detections"] = detections[offset : offset + limit]
    result["total"] = len(detections)
    result["limit"] = limit
    result["offset"] = offset
    return JSONResponse(result)


@app.get("/v1/export/{task_id}", response_model=None)
async def export_csv(
    task_id: str,
    api_key: str = Depends(verify_api_key),
) -> FileResponse | JSONResponse:
    """Export batch results as CSV."""
    if task_id not in batch_results:
        return JSONResponse({"error": "Task not found"}, status_code=404)

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
