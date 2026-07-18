"""Service layer for detection business logic."""

import asyncio
import csv
import hashlib
import io
import logging
import random
import re
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .utils import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_VIDEO_SIZE,
    validate_image,
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


async def retry_with_backoff(
    func, *args,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    **kwargs,
):
    """Retry an async function with exponential backoff and jitter.

    Args:
        func: Async callable to retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except (OSError, ConnectionError, TimeoutError) as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                # Add jitter: ±25% of delay
                jitter = delay * 0.25 * (2 * random.random() - 1)
                await asyncio.sleep(delay + jitter)
            else:
                raise last_exc


class CircuitBreaker:
    """Simple circuit breaker to prevent cascading failures.

    Tracks consecutive failures and opens the circuit when a threshold
    is reached, allowing the system to recover before retrying.
    """

    OPEN = "open"
    HALF_OPEN = "half-open"
    CLOSED = "closed"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = self.CLOSED
        self.last_failure_time = 0.0

    def call(self, func, *args, **kwargs):
        """Execute func if circuit is closed, raise CircuitBreakerError if open."""
        if self.state == self.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = self.HALF_OPEN
            else:
                raise RuntimeError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                self.failure_count = 0
            return result
        except Exception as exc:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
            raise exc

    async def async_call(self, func, *args, **kwargs):
        """Async version of call()."""
        if self.state == self.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = self.HALF_OPEN
            else:
                raise RuntimeError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                self.failure_count = 0
            return result
        except Exception as exc:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
            raise exc


class DetectionCache:
    """Simple LRU cache for detection results to avoid redundant inference (PER-04)."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()

    def _make_key(self, image_path: str) -> str:
        return f"path:{image_path}"

    def _make_array_key(self, img: np.ndarray) -> str:
        # Use first 32 bytes of content as key (fast, not full hash)
        return f"arr:{hashlib.md5(img.tobytes()[:4096]).hexdigest()}"

    def get(self, key: str) -> list | None:
        """Get cached result and mark as recently used."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: list) -> None:
        """Store result in cache, evicting oldest if at capacity."""
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()


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

    # ---- File helpers ----

    def get_upload_path(self, filename: str) -> Path:
        """Generate a unique upload path for a file."""
        safe_name = sanitize_filename(filename)
        return self.upload_dir / f"{uuid.uuid4().hex}_{safe_name}"

    def save_upload(self, contents: bytes, path: Path) -> None:
        """Save uploaded file contents to disk."""
        path.write_bytes(contents)

    # ---- Validation ----

    def validate_image(self, filename: str, size: int) -> None:
        """Validate image extension and size (raises HTTPException on failure)."""
        validate_image(filename, size)

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
        """Run detection on a single image (thread-pooled with cache)."""
        import asyncio

        cache_key = self.cache._make_key(image_path)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        result = await asyncio.wait_for(
            asyncio.to_thread(self.detector.detect, image_path),
            timeout=30.0,
        )
        self.cache.put(cache_key, result)
        return result

    async def run_detection_on_array(self, img: np.ndarray) -> list[dict]:
        """Run detection on a numpy array (thread-pooled)."""
        import asyncio

        return await asyncio.wait_for(
            asyncio.to_thread(self.detector.detect_array, img),
            timeout=30.0,
        )

    async def run_detection_on_paths(self, file_paths: list[str]) -> list[dict]:
        """Run detection on multiple image paths (thread-pooled)."""
        import asyncio

        return await asyncio.wait_for(
            asyncio.to_thread(self.pipeline.process, file_paths),
            timeout=300.0,
        )

    # ---- Batch processing ----

    async def process_batch(self, task_id: str, file_paths: list[str]) -> None:
        """Background task: run detection and store results."""
        logger.info("Batch %s: processing %d files", task_id, len(file_paths))
        try:
            results = await self.run_detection_on_paths(file_paths)
            aggregated = self.pipeline.aggregate(results)
            computed_stats = self.pipeline.stats(aggregated)

            self._evict_oldest_batch()

            self.batch_results[task_id] = {
                "files_processed": len(file_paths),
                "detections": aggregated,
                "per_file": results,
                "stats": computed_stats,
                "_ts": time.time(),
            }
            logger.info("Batch %s: completed (%d objects)", task_id, len(aggregated))
        except (ValueError, RuntimeError, OSError) as exc:
            logger.error("Batch %s failed: %s", task_id, exc)

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

        csv_path = self.output_dir / f"{task_id}.csv"
        csv_path.write_text(output.getvalue())
        return str(csv_path)
