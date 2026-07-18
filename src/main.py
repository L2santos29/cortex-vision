"""Cortex-Vision main entry point — FastAPI application."""

import asyncio
import logging
import statistics
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from pydantic_settings import BaseSettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .detector import Detector
from .pipeline import BatchPipeline
from .services import DetectionService, sanitize_filename
from .utils import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_VIDEO_SIZE,
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
# ---- HTTPS startup check (SEC-10) ----
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Application lifespan — startup/shutdown hooks."""
    logger.warning(
        "Cortex-Vision is starting without HTTPS. In production, "
        "deploy behind a TLS-terminating reverse proxy (nginx, Caddy, Traefik)."
    )
    yield


app = FastAPI(title="Cortex-Vision", version="0.1.0", lifespan=lifespan)

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


async def verify_api_key(
    api_key: str = Security(api_key_header),
    request: Request = None,
) -> str:
    """Dependency that validates API key for non-public endpoints.

    Checks the ``X-API-Key`` header first, then falls back to the ``api_key``
    HTTP-only cookie (set by the index page for the web UI).
    """
    if api_key == settings.api_key:
        return api_key
    # Fallback: check the HTTP-only cookie (set by the web UI)
    if request is not None:
        cookie_key = request.cookies.get("api_key")
        if cookie_key == settings.api_key:
            return cookie_key
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


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
        if request.url.path in ("/health", "/", "/v1/detect/frame"):
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

# ---- Metrics (MON-03) ----
REQUEST_COUNT: dict[str, int] = defaultdict(int)
REQUEST_LATENCY: dict[str, list[float]] = defaultdict(list)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        route = request.url.path
        response = await call_next(request)
        latency = time.time() - start
        REQUEST_COUNT[route] += 1
        REQUEST_LATENCY[route].append(latency)
        # Keep only last 1000 latency measurements per route
        if len(REQUEST_LATENCY[route]) > 1000:
            REQUEST_LATENCY[route] = REQUEST_LATENCY[route][-1000:]
        return response


app.add_middleware(MetricsMiddleware)


@app.get("/metrics")
async def metrics(api_key: str = Depends(verify_api_key)):
    """Prometheus-compatible metrics endpoint."""
    lines = [
        "# HELP cortex_vision_request_count Total request count by route",
        "# TYPE cortex_vision_request_count counter",
    ]
    for route, count in sorted(REQUEST_COUNT.items()):
        lines.append(f'cortex_vision_request_count{{route="{route}"}} {count}')

    lines.append("# HELP cortex_vision_request_latency_seconds Request latency by route")
    lines.append("# TYPE cortex_vision_request_latency_seconds gauge")
    for route, latencies in sorted(REQUEST_LATENCY.items()):
        if latencies:
            avg = statistics.mean(latencies)
            lines.append(
                f'cortex_vision_request_latency_seconds{{route="{route}",quantile="avg"}} {avg:.4f}'
            )
            if len(latencies) >= 2:
                p99 = sorted(latencies)[int(len(latencies) * 0.99)]
                lines.append(
                    f'cortex_vision_request_latency_seconds{{route="{route}",quantile="p99"}} {p99:.4f}'
                )

    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


# ---- Engine ----
detector = Detector(model_name=settings.yolo_model)
pipeline = BatchPipeline(detector)

# ---- Service layer ----
service = DetectionService(
    detector=detector,
    pipeline=pipeline,
    upload_dir=settings.upload_dir,
    output_dir=settings.output_dir,
)

# ---- Named constants ----
DEFAULT_PAGINATION_LIMIT = 100

# Serve static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Routes — public (no auth) at root level (ARC-09)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> Response:
    """Serve the web UI (public, no auth required).

    Sets an HTTP-only cookie with the API key so the frontend can
    authenticate without exposing the key to client-side JavaScript.
    """
    index_path = static_dir / "index.html"
    if index_path.exists():
        html = index_path.read_text()
        response = HTMLResponse(content=html)
        response.set_cookie(
            key="api_key",
            value=settings.api_key,
            httponly=True,
            samesite="lax",
            max_age=86400,  # 24 hours
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
        service.save_upload(contents, image_path)
    except OSError as exc:
        logger.error("Failed to write uploaded image %s: %s", image_path, exc)
        return JSONResponse({"error": "Failed to save uploaded file"}, status_code=500)

    try:
        results = await service.run_detection(str(image_path))
    except (ValueError, RuntimeError, OSError) as exc:
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
            service.save_upload(contents, image_path)
        except OSError as exc:
            logger.error("Failed to write batch file %s: %s", image_path, exc)
            continue

        file_paths.append(str(image_path))

    # Schedule background processing (PER-02)
    background_tasks.add_task(service.process_batch, task_id, file_paths)

    return JSONResponse(
        {
            "task_id": task_id,
            "files_processed": len(file_paths),
        },
        status_code=201,
    )


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

    contents = await file.read()

    if len(contents) > MAX_VIDEO_SIZE:
        return JSONResponse({"error": "Video too large (>500MB)"}, status_code=400)

    # Validate video content via magic bytes
    try:
        from .utils import validate_video_content
        validate_video_content(contents)
    except HTTPException:
        raise

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
    """Receive a single frame from webcam and run detection.

    Lightweight endpoint — no disk I/O, processes entirely in memory.
    Called periodically (~1 FPS) by the live webcam UI.
    """
    contents = await file.read()
    if not contents:
        return JSONResponse({"error": "Empty frame"}, status_code=400)

    # Validate size (SEC-01)
    MAX_FRAME_SIZE = 5 * 1024 * 1024
    if len(contents) > MAX_FRAME_SIZE:
        return JSONResponse({"error": "Frame too large (>5MB)"}, status_code=400)

    # Decode frame — cv2 validates the actual image data (Content-Type header from
    # FormData blobs is unreliable across browsers; real validation is in imdecode)
    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error": "Invalid image data"}, status_code=400)
    except Exception as exc:
        logger.error("Frame decode failed: %s", exc)
        return JSONResponse({"error": "Failed to decode frame"}, status_code=400)

    # Run detection with error handling (ERR-01)
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
