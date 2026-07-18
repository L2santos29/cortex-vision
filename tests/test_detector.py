"""Tests for the YOLOv8 detector module."""

import numpy as np
import pytest

from src.detector import Detector


def test_detector_init_default_model():
    """Detector initializes with default model name."""
    detector = Detector()
    assert detector.model_name == "yolov8n.pt"
    assert detector._model is None  # Lazy loading


def test_detector_custom_model():
    """Detector accepts custom model name."""
    detector = Detector(model_name="yolov8s.pt")
    assert detector.model_name == "yolov8s.pt"


def test_detector_model_lazy_load(mock_yolo):
    """Model property triggers YOLO loading only on first access."""
    detector = Detector()
    assert detector._model is None

    # Access model property — triggers lazy load
    m = detector.model
    assert detector._model is not None
    assert m is detector._model

    # Second access returns cached model
    assert detector.model is m


def test_detect_array_valid_input(mock_yolo):
    """detect_array returns correctly formatted detections."""
    from tests.conftest import MockResults

    mock_yolo.return_value = [
        MockResults(
            names={0: "person", 2: "car"},
            boxes_data=[
                (0, 0.95, [10.0, 20.0, 100.0, 200.0]),
                (2, 0.85, [50.0, 60.0, 300.0, 150.0]),
            ],
        )
    ]

    detector = Detector()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect_array(img)

    assert len(detections) == 2
    assert detections[0]["class"] == "person"
    assert detections[0]["confidence"] == 0.95
    assert detections[0]["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert detections[1]["class"] == "car"
    assert detections[1]["confidence"] == 0.85
    assert detections[1]["bbox"] == [50.0, 60.0, 300.0, 150.0]


def test_detect_returns_empty_list_when_no_boxes(mock_yolo, monkeypatch):
    """detect returns empty list when YOLO finds no objects."""
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

    from tests.conftest import MockResults

    mock_yolo.return_value = [
        MockResults(names={0: "person"}, boxes_data=None)
    ]

    detector = Detector()
    detections = detector.detect("empty.jpg")
    assert detections == []


def test_detect_returns_correct_keys(mock_yolo, monkeypatch):
    """Each detection dict contains the expected keys."""
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

    from tests.conftest import MockResults

    mock_yolo.return_value = [
        MockResults(
            names={0: "person"},
            boxes_data=[(0, 0.95, [10.0, 20.0, 100.0, 200.0])],
        )
    ]

    detector = Detector()
    detections = detector.detect("test.jpg")

    assert len(detections) == 1
    det = detections[0]
    assert list(det.keys()) == ["class", "confidence", "bbox"]


def test_detect_array_handles_empty_array(mock_yolo):
    """detect_array handles an empty numpy array without crashing."""
    from tests.conftest import MockResults

    mock_yolo.return_value = [
        MockResults(names={0: "person"}, boxes_data=None)
    ]

    detector = Detector()
    empty_img = np.empty((0, 0, 3), dtype=np.uint8)
    detections = detector.detect_array(empty_img)
    assert detections == []
