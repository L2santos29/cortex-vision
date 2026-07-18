"""Utility functions for image preprocessing and helpers."""

from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


def validate_image(filename: str, size: int) -> None:
    """Validate uploaded image file.

    Args:
        filename: Original filename.
        size: File size in bytes.

    Raises:
        HTTPException: If validation fails.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    if size > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large (>10MB)")


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB


def extract_frames(video_path: str, fps_sample: int = 1) -> list[dict]:
    """Extract frames from a video at a given sampling rate.

    Args:
        video_path: Path to the video file.
        fps_sample: Number of frames per second to extract (default 1).

    Returns:
        List of dicts with 'frame' (ndarray), 'timestamp' (float), 'frame_number' (int).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30  # fallback

    frame_interval = max(1, int(round(video_fps / fps_sample)))
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            frames.append({
                "frame": frame,
                "timestamp": round(timestamp, 1),
                "frame_number": frame_idx,
            })
        frame_idx += 1

    cap.release()
    return frames


# Reuse the existing model's predict method to avoid circular imports
# Type hint using string literal to avoid import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .detector import Detector


def process_video_frames(video_path: str, detector: "Detector", fps_sample: int = 1) -> list[dict]:
    """Extract frames from a video and run detection on each.

    Args:
        video_path: Path to the video file.
        detector: Detector instance.
        fps_sample: Frames per second to sample.

    Returns:
        List of dicts with timestamp, frame_number, detections, object_count.
    """
    frames = extract_frames(video_path, fps_sample)
    results = []

    for f in frames:
        # Write frame to temp file for detection
        temp_path = f"/tmp/{Path(video_path).stem}_frame{f['frame_number']}.jpg"
        cv2.imwrite(temp_path, f["frame"])
        detections = detector.detect(temp_path)
        Path(temp_path).unlink(missing_ok=True)

        results.append({
            "timestamp": f["timestamp"],
            "frame_number": f["frame_number"],
            "detections": detections,
            "object_count": len(detections),
        })

    return results


def preprocess(image_path: str, target_size: tuple[int, int] = (640, 640)) -> np.ndarray:
    """Load and preprocess an image for inference.

    Args:
        image_path: Path to the image.
        target_size: Desired output size (width, height).

    Returns:
        Preprocessed image as numpy array (BGR).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, target_size)
    return img
