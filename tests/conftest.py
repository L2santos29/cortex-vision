"""Reusable fixtures and mock helpers for cortex-vision tests."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class MockBox:
    """Mock for a single YOLO bounding box."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = np.array(cls_id)
        self.conf = np.array(conf)
        self.xyxy = np.array([xyxy])  # shape (1, 4)


class MockBoxes:
    """Mock for YOLO Boxes container — iterable of MockBox."""

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
def _api_key_env(monkeypatch):
    """Ensure API_KEY is set so src.main can be imported."""
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
    """Patch ultralytics.YOLO and return the mock instance.

    Tests configure detection results by setting
    ``mock_yolo.return_value`` to a list of MockResults.
    """
    with patch("src.detector.YOLO") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance
