"""Tests for the YOLOv8 detector module."""

import pytest
from src.detector import Detector


def test_detector_init():
    """Detector initializes with default model."""
    detector = Detector()
    assert detector.model_name == "yolov8n.pt"
    assert detector._model is None  # Lazy loading


def test_detector_custom_model():
    """Detector accepts custom model name."""
    detector = Detector(model_name="yolov8s.pt")
    assert detector.model_name == "yolov8s.pt"
