"""Shared fixtures and helpers for all test categories."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class MockBox:
    """Mock for a single YOLO bounding box."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = np.array(cls_id)
        self.conf = np.array(conf)
        self.xyxy = np.array([xyxy])

    def item(self):
        return int(self.cls[0])


class MockBoxes:
    """Mock for YOLO Boxes container."""

    def __init__(self, boxes_data):
        self._boxes = [MockBox(*b) for b in (boxes_data or [])]

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class MockResults:
    """Mock for a single YOLO prediction result."""

    def __init__(self, names=None, boxes_data=None):
        self.names = names or {0: "person"}
        self.boxes = MockBoxes(boxes_data) if boxes_data else None


@pytest.fixture(autouse=True)
def api_key_env(monkeypatch):
    """Ensure API_KEY is set so src modules can be imported."""
    monkeypatch.setenv("API_KEY", "test-key-for-testing")


@pytest.fixture
def sample_detections():
    """Sample detection dicts for testing aggregation and stats."""
    return [
        {"class": "person", "confidence": 0.95, "bbox": [10.0, 20.0, 100.0, 200.0]},
        {"class": "car", "confidence": 0.85, "bbox": [50.0, 60.0, 300.0, 150.0]},
        {"class": "dog", "confidence": 0.72, "bbox": [200.0, 300.0, 400.0, 450.0]},
    ]


@pytest.fixture
def mock_yolo():
    """Patch ultralytics.YOLO and return the mock instance."""
    with patch("src.detector.YOLO") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_detector():
    """Create a MockDetector with controllable return values."""
    detector = MagicMock()
    detector.model_name = "yolov8n.pt"
    detector.detect.return_value = []
    detector.detect_array.return_value = []
    return detector


@pytest.fixture
def mock_pipeline():
    """Create a mock BatchPipeline."""
    pipeline = MagicMock()
    pipeline.process.return_value = []
    pipeline.aggregate.return_value = []
    pipeline.stats.return_value = {
        "total_detections": 0,
        "unique_classes": 0,
        "per_class": {},
        "top_classes": [],
    }
    return pipeline


@pytest.fixture
def sample_image_bytes():
    """Return minimal valid JPEG bytes for testing uploads."""
    # Minimal valid JPEG (SOI + EOI markers) — not decodeable but passes magic-byte check
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\x00" * 100


@pytest.fixture
def sample_video_bytes():
    """Return minimal valid MP4-like bytes for testing."""
    # ftyp box prefix — passes magic-byte check
    return b"\x00\x00\x00\x1cftypmp42\x00\x00\x00\x00mp42mp41" + b"\x00" * 100


@pytest.fixture
def sample_np_image():
    """Return a small numpy array simulating an image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)
