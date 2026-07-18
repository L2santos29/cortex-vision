"""YOLOv8 object detection engine."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO


class Detector:
    """Object detector using YOLOv8.

    Handles model loading, inference, and result formatting.
    """

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        """Initialize detector with a YOLOv8 model variant.

        Args:
            model_name: YOLOv8 model variant (n, s, m, l, x).
                        Defaults to 'yolov8n.pt' (nano, fastest).
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
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

    def detect(self, image_path: str) -> list[dict]:
        """Run object detection on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of detection dicts with keys: class, confidence, bbox.
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        results = self.model(image_path, verbose=False)
        return self._parse_results(results)

    def detect_array(self, img: np.ndarray) -> list[dict]:
        """Run object detection on a numpy image array (no disk I/O).

        Args:
            img: Image as numpy array (BGR format from OpenCV).

        Returns:
            List of detection dicts with keys: class, confidence, bbox.
        """
        results = self.model(img, verbose=False)
        return self._parse_results(results)
