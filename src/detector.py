"""YOLOv8 object detection engine."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .profiling import timed


class Detector:
    """Object detector using YOLOv8.

    Handles model loading, inference, and result formatting.
    """

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        """Configure detector with a YOLOv8 model variant.

        The model file is downloaded automatically by Ultralytics on first use
        if not present locally. Lazy-loading defers download until the first
        detect/detect_array call.

        Args:
            model_name: YOLOv8 model variant. Nano (n) is fastest; use
                        s/m/l/x for better accuracy at the cost of speed.
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> YOLO:
        """Lazy-load the YOLO model."""
        if self._model is None:
            self._model = YOLO(self.model_name)
        return self._model

    def _parse_results(self, results) -> list[dict]:
        """Extract detection dicts from YOLO results (ARC-06)."""
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    class_name = result.names[cls_id]
                    confidence = float(box.conf.item())
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    detections.append({
                        "class": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": [round(x, 1) for x in bbox],
                    })
        return detections

    @timed(warn_threshold=5.0)
    def detect(self, image_path: str) -> list[dict]:
        """Run object detection on a single image file.

        Reads the image from disk, runs YOLO inference, and returns formatted
        results. Use detect_array() instead when the image is already in memory
        to avoid disk I/O.

        Raises:
            FileNotFoundError: If the image path does not exist.
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            results = self.model(image_path, verbose=False)
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed on {image_path}") from exc
        return self._parse_results(results)

    @timed(warn_threshold=5.0)
    def detect_array(self, img: np.ndarray) -> list[dict]:
        """Run object detection on a numpy image array (no disk I/O).

        Useful for webcam frames or in-memory images where writing to disk
        would add latency. Expects BGR format from OpenCV.
        """
        try:
            results = self.model(img, verbose=False)
        except Exception as exc:
            raise RuntimeError("YOLO inference failed on array input") from exc
        return self._parse_results(results)
