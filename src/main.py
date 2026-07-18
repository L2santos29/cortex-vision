"""Cortex-Vision main entry point — FastAPI application."""

import asyncio
import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import settings
from .detector import Detector
from .middleware import (
    HTTPSRedirectMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    metrics_endpoint,
    verify_api_key,
)
from .pipeline import BatchPipeline
from .services import DetectionService
from .utils import ALLOWED_VIDEO_EXTENSIONS, MAX_VIDEO_SIZE

# ---- Named constants (STY-08) ----
COOKIE_MAX_AGE = 86400  # 24 hours in seconds
MAX_FRAME_SIZE = 5 * 1024 * 1024  # 5 MB

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Cortex-Vision", version="0.1.0")

# Compression (PER-07)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# HTTPS redirect (SEC-10)
app.add_middleware(HTTPSRedirectMiddleware)

# CORS (SEC-04)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type", "Accept"],
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(MetricsMiddleware)

# ---- Engine & Service ----
detector = Detector(model_name=settings.yolo_model)
pipeline = BatchPipeline(detector)
service = DetectionService(
    detector=detector,
    pipeline=pipeline,
    upload_dir=settings.upload_dir,
    output_dir=settings.output_dir,
)

DEFAULT_PAGINATION_LIMIT = 100

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ---- Metrics ----
@app.get("/metrics")
async def metrics(api_key: str = Depends(verify_api_key)):
    return await metrics_endpoint()

# ---------------------------------------------------------------------------
# Routes — public (no auth)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> Response:
    """Serve the web UI (public, no auth required)."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        html = index_path.read_text()
        response = HTMLResponse(content=html)
        response.set_cookie(
            key="api_key",
            value=settings.api_key,
            httponly=True,
            samesite="lax",
            max_age=COOKIE_MAX_AGE,
        )
        return response
    return HTMLResponse(
        content="<h1>Cortex-Vision API</h1><p>Static UI not found. Use /docs for API.</p>"
    )


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
    service.validate_image(file.filename or "unknown", len(contents), contents)
    image_path = service.get_upload_path(file.filename or "unknown")

    try:
        await service.save_upload(contents, image_path)
    except OSError as exc:
        logger.error("Failed to write uploaded image %s: %s", image_path, exc)
        return JSONResponse({"error": "Failed to save uploaded file"}, status_code=500)

    try:
        results = await service.run_detection(str(image_path))
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
        service.validate_image(file.filename or "unknown", len(contents), contents)
        image_path = service.get_batch_upload_path(task_id, file.filename or "unknown")

        try:
            await service.save_upload(contents, image_path)
        except OSError as exc:
            logger.error("Failed to write batch file %s: %s", image_path, exc)
            continue

        file_paths.append(str(image_path))

    background_tasks.add_task(service.process_batch, task_id, file_paths)
    return JSONResponse(
        {"task_id": task_id, "files_processed": len(file_paths)},
        status_code=201,
    )


@app.post("/v1/upload/video", status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
) -> JSONResponse:
    """Upload a video and get per-frame detection results."""
    filename = file.filename or "unknown"
    if not service.validate_video_format(filename):
        return JSONResponse(
            {
                "error": (
                    f"Unsupported video format. Allowed: "
                    f"{', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
                )
            },
            status_code=400,
        )

    contents = await file.read()
    if not service.validate_video_size(len(contents)):
        return JSONResponse({"error": "Video too large (>500MB)"}, status_code=400)

    from .utils import validate_video_content
    validate_video_content(contents)

    video_id = uuid.uuid4().hex

    try:
        result = await service.process_video(contents, file.filename or "unknown", video_id)
        logger.info(
            "Processed video %s (%d frames)",
            file.filename,
            result["total_frames_processed"],
        )
        return JSONResponse(result, status_code=201)
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Video processing failed for %s: %s", file.filename, exc)
        return JSONResponse({"error": "Video processing failed"}, status_code=500)
    except Exception as exc:
        logger.error("Unexpected error processing video %s: %s", file.filename, exc)
        return JSONResponse({"error": "Video processing failed"}, status_code=500)


@app.post("/v1/detect/frame")
async def detect_frame(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
) -> JSONResponse:
    """Receive a single frame from webcam and run detection (no disk I/O)."""
    contents = await file.read()
    if not contents:
        return JSONResponse({"error": "Empty frame"}, status_code=400)

    if len(contents) > MAX_FRAME_SIZE:
        return JSONResponse({"error": "Frame too large (>5MB)"}, status_code=400)

    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Invalid image data"}, status_code=400)
    except Exception as exc:
        logger.error("Frame decode failed: %s", exc)
        return JSONResponse({"error": "Failed to decode frame"}, status_code=400)

    try:
        detections = await service.run_detection_on_array(img)
    except asyncio.TimeoutError:
        logger.error("Frame detection timed out")
        return JSONResponse({"error": "Detection timed out"}, status_code=504)
    except Exception as exc:
        logger.error("Frame detection failed: %s", exc)
        return JSONResponse({"error": "Detection failed"}, status_code=500)

    return JSONResponse({
        "detections": detections,
        "object_count": len(detections),
        "timestamp": time.time(),
    })


@app.get("/v1/results/{task_id}")
async def get_results(
    task_id: str,
    limit: int = DEFAULT_PAGINATION_LIMIT,
    offset: int = 0,
    api_key: str = Depends(verify_api_key),
) -> JSONResponse:
    """Get batch processing results with pagination (PER-03)."""
    result = service.get_batch_result(task_id)
    if result is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    result = dict(result)
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
    csv_path = service.export_batch_csv(task_id)
    if csv_path is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return FileResponse(csv_path, filename=f"cortex-vision-{task_id}.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
