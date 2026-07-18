"""Cortex-Vision — Object Recognition Platform.

Powered by YOLOv8. Supports images, video, and webcam detection.

Quick start:
    from src import Detector

    detector = Detector()
    results = detector.detect("image.jpg")
    print(results)

Or via CLI:
    cortex-vision detect image.jpg
"""

from .detector import Detector
from .pipeline import BatchPipeline
from .services import DetectionService, sanitize_filename
from .utils import process_video_frames, extract_frames

__all__ = [
    "Detector",
    "BatchPipeline",
    "DetectionService",
    "sanitize_filename",
    "process_video_frames",
    "extract_frames",
]
