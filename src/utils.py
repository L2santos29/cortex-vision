"""Utility functions for image and video processing."""

import base64
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from fastapi import HTTPException

if TYPE_CHECKING:
    from .detector import Detector

# ---- Named constants (STY-08) ----
THUMBNAIL_MAX_SIZE = 320
JPEG_QUALITY = 75
HUE_RANGE = 120
FALLBACK_FPS = 30

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
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to open video: {video_path}") from exc

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = FALLBACK_FPS

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


def draw_boxes_on_frame(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes on a frame using OpenCV.

    Color transitions smoothly from red (0% confidence) through yellow to green (100%).

    Args:
        frame: Image as numpy array (BGR).
        detections: List of detection dicts with bbox, class, confidence.

    Returns:
        Frame with boxes drawn (BGR).
    """
    h, w = frame.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        conf = d["confidence"]

        # Smooth color: red (hue=0) → yellow (hue=60) → green (hue=120)
        hue = int(conf * HUE_RANGE)
        color_rgb = cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
        )[0][0]
        color = (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))

        # Clamp to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Bounding box rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{d['class']} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 22, 0)
        cv2.rectangle(frame, (x1, label_y), (x1 + tw + 8, label_y + 20), color, -1)

        # Label text
        cv2.putText(
            frame, label, (x1 + 4, label_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    return frame


def frame_to_base64(frame: np.ndarray, max_size: int = THUMBNAIL_MAX_SIZE) -> str:
    """Encode a frame as a base64 JPEG thumbnail.

    Args:
        frame: Image as numpy array (BGR).
        max_size: Maximum dimension for the thumbnail.

    Returns:
        Base64-encoded JPEG string with 'data:image/jpeg;base64,' prefix.
    """
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def process_video_frames(video_path: str, detector: "Detector", fps_sample: int = 1) -> list[dict]:
    """Extract frames from a video, run detection, and return annotated images.

    For each frame: runs YOLO detection, draws bounding boxes on the frame,
    and encodes the result as a base64 JPEG thumbnail.

    Args:
        video_path: Path to the video file.
        detector: Detector instance.
        fps_sample: Frames per second to sample.

    Returns:
        List of dicts with timestamp, frame_number, detections, object_count,
        annotated_frame (base64 JPEG data URL).
    """
    frames = extract_frames(video_path, fps_sample)
    results = []

    for f in frames:
        frame = f["frame"]

        # Run detection directly on array (no disk I/O)
        detections = detector.detect_array(frame)

        # Draw bounding boxes on the frame
        annotated = draw_boxes_on_frame(frame.copy(), detections)

        # Encode as base64 thumbnail
        frame_b64 = frame_to_base64(annotated)

        results.append({
            "timestamp": f["timestamp"],
            "frame_number": f["frame_number"],
            "detections": detections,
            "object_count": len(detections),
            "annotated_frame": frame_b64,
        })

    return results



