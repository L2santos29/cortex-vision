"""Service layer for detection business logic."""

import asyncio
import csv
import io
import logging
import re
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .cache import DetectionCache
from .resilience import CircuitBreaker, retry_with_backoff
from .utils import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_VIDEO_SIZE,
    process_video_frames,
    validate_image,
    validate_image_content,
    validate_video_content,
)

if TYPE_CHECKING:
    from .detector import Detector
    from .pipeline import BatchPipeline

logger = logging.getLogger(__name__)

# ---- Constants ----
MAX_FILENAME_LENGTH = 128
MAX_BATCH_RESULTS = 1000


def sanitize_filename(filename: str) -> str:
    """Remove path separators and dangerous characters from a filename."""
    safe = Path(filename).name
    safe = re.sub(r"[^\w.\- ]", "_", safe)
    return safe[:MAX_FILENAME_LENGTH]


class DetectionService:
    """Encapsulates detection business logic separating it from HTTP concerns."""

    def __init__(
        self,
        detector: "Detector",
        pipeline: "BatchPipeline",
        upload_dir: str = "uploads",
        output_dir: str = "output",
    ) -> None:
        self.detector = detector
        self.pipeline = pipeline
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.cache = DetectionCache()
        self.batch_results: dict[str, dict] = {}
        self.detection_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

    # ---- File helpers ----

    def get_upload_path(self, filename: str) -> Path:
        """Generate a unique upload path for a file."""
        safe_name = sanitize_filename(filename)
        return self.upload_dir / f"{uuid.uuid4().hex}_{safe_name}"

    def get_batch_upload_path(self, prefix: str, filename: str) -> Path:
        """Generate an upload path using a prefix (e.g. task_id) for batch files.

        The prefix is visible in the filename so the frontend can correlate
        results back to the original upload without knowing the random UUID.
        """
        safe_name = sanitize_filename(filename)
        return self.upload_dir / f"{prefix}_{safe_name}"

    async def save_upload(self, contents: bytes, path: Path) -> None:
        """Save uploaded file contents to disk with retry on transient I/O errors."""
        async def _write(p: Path, c: bytes) -> None:
            p.write_bytes(c)
        await retry_with_backoff(
            _write, path, contents,
            max_retries=2,
            base_delay=0.2,
        )

    # ---- Validation ----

    def validate_image(self, filename: str, size: int, contents: bytes | None = None) -> None:
        """Validate image extension, size, and content (raises HTTPException on failure)."""
        validate_image(filename, size)
        if contents is not None:
            validate_image_content(contents)

    def validate_video_format(self, filename: str) -> bool:
        """Check if the video format is supported."""
        safe_name = sanitize_filename(filename)
        ext = Path(safe_name).suffix.lower()
        return ext in ALLOWED_VIDEO_EXTENSIONS

    def validate_video_size(self, size: int) -> bool:
        """Check if the video size is within limits."""
        return size <= MAX_VIDEO_SIZE

    # ---- Detection ----

    async def run_detection(self, image_path: str) -> list[dict]:
        """Run detection on a single image (thread-pooled with cache and circuit breaker)."""
        import asyncio

        cache_key = self.cache._make_key(image_path)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        async def _detect():
            return await asyncio.wait_for(
                asyncio.to_thread(self.detector.detect, image_path),
                timeout=30.0,
            )

        result = await self.detection_circuit.async_call(_detect)
        self.cache.put(cache_key, result)
        return result

    async def run_detection_on_array(self, img: np.ndarray) -> list[dict]:
        """Run detection on a numpy array (thread-pooled, circuit-protected)."""
        import asyncio

        async def _detect():
            return await asyncio.wait_for(
                asyncio.to_thread(self.detector.detect_array, img),
                timeout=30.0,
            )

        return await self.detection_circuit.async_call(_detect)

    async def run_detection_on_paths(self, file_paths: list[str]) -> list[dict]:
        """Run detection on multiple image paths (thread-pooled, circuit-protected)."""
        import asyncio

        async def _detect():
            return await asyncio.wait_for(
                asyncio.to_thread(self.pipeline.process, file_paths),
                timeout=300.0,
            )

        return await self.detection_circuit.async_call(_detect)

    # ---- Video processing ----

    async def process_video(
        self, contents: bytes, filename: str, video_id: str
    ) -> dict:
        """Save a video, run per-frame detection, and return annotated results.

        The temporary video file is cleaned up after processing. All detection
        runs on the YOLO engine via the shared detector instance.

        Returns:
            A dict with keys: filename, video_id, duration_seconds,
            total_frames_processed, total_objects, frames.
        """
        safe_name = sanitize_filename(filename)
        video_path = self.upload_dir / f"{video_id}_{safe_name}"

        # Save to disk with retry on transient I/O errors
        async def _write_video(p: Path, c: bytes) -> None:
            p.write_bytes(c)
        await retry_with_backoff(
            _write_video, video_path, contents,
        )

        try:
            frames_data = await asyncio.to_thread(
                process_video_frames, str(video_path), self.detector, 1
            )
            duration = frames_data[-1]["timestamp"] if frames_data else 0

            return {
                "filename": filename,
                "video_id": video_id,
                "duration_seconds": duration,
                "total_frames_processed": len(frames_data),
                "total_objects": sum(f["object_count"] for f in frames_data),
                "frames": frames_data,
            }
        finally:
            if video_path.exists():
                video_path.unlink()

    # ---- Batch processing ----

    async def process_batch(self, task_id: str, file_paths: list[str]) -> None:
        """Background task: run detection and store results (graceful degradation).

        If some images fail detection, the batch still stores partial results
        with error markers instead of failing entirely.
        """
        logger.info("Batch %s: processing %d files", task_id, len(file_paths))
        try:
            results = await self.run_detection_on_paths(file_paths)
        except Exception as exc:
            logger.error("Batch %s completely failed: %s", task_id, exc)
            self._evict_oldest_batch()
            self.batch_results[task_id] = {
                "files_processed": len(file_paths),
                "detections": [],
                "per_file": [],
                "stats": {"total_detections": 0, "unique_classes": 0, "per_class": {}, "top_classes": []},
                "errors": [str(exc)],
                "_ts": time.time(),
            }
            return

        failed = sum(1 for r in results if "error" in r)
        aggregated = self.pipeline.aggregate(results)
        computed_stats = self.pipeline.stats(aggregated)

        self._evict_oldest_batch()

        self.batch_results[task_id] = {
            "files_processed": len(file_paths),
            "files_failed": failed,
            "detections": aggregated,
            "per_file": results,
            "stats": computed_stats,
            "_ts": time.time(),
        }

        if failed:
            logger.warning(
                "Batch %s: completed with %d/%d files failed (%d objects)",
                task_id, failed, len(file_paths), len(aggregated),
            )
        else:
            logger.info("Batch %s: completed (%d objects)", task_id, len(aggregated))

    def _evict_oldest_batch(self) -> None:
        """Remove oldest entry when batch_results exceeds the limit."""
        if len(self.batch_results) >= MAX_BATCH_RESULTS:
            oldest = min(
                self.batch_results.keys(),
                key=lambda k: self.batch_results[k].get("_ts", 0),
            )
            del self.batch_results[oldest]

    # ---- Results ----

    def get_batch_result(self, task_id: str) -> dict | None:
        """Get batch results for a task ID."""
        return self.batch_results.get(task_id)

    def export_batch_csv(self, task_id: str) -> str | None:
        """Export batch results as CSV and return the file path."""
        result = self.get_batch_result(task_id)
        if result is None:
            return None

        detections = result["detections"]
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

        safe_name = sanitize_filename(f"{task_id}.csv")
        csv_path = self.output_dir / safe_name
        csv_path.write_text(output.getvalue())
        return str(csv_path)
