"""Tests for the service layer."""
from unittest.mock import MagicMock

import pytest

from src.services import sanitize_filename, DetectionService


def test_sanitize_filename_removes_path():
    """sanitize_filename strips directory components."""
    assert sanitize_filename("/etc/passwd") == "passwd"
    assert sanitize_filename("../../etc/passwd") == "passwd"


def test_sanitize_filename_removes_special_chars():
    """sanitize_filename replaces special chars with underscores."""
    result = sanitize_filename("hello$%^world.txt")
    assert "$" not in result
    assert "%" not in result
    assert "^" not in result


def test_sanitize_filename_limits_length():
    """sanitize_filename truncates to MAX_FILENAME_LENGTH."""
    long_name = "a" * 200 + ".txt"
    result = sanitize_filename(long_name)
    assert len(result) <= 128


def test_detection_service_init():
    """DetectionService initializes with detector and pipeline."""
    detector = MagicMock()
    pipeline = MagicMock()
    service = DetectionService(detector, pipeline)
    assert service.detector is detector
    assert service.pipeline is pipeline


def test_get_upload_path_generates_unique_path():
    """get_upload_path returns a unique path with a UUID."""
    detector = MagicMock()
    pipeline = MagicMock()
    service = DetectionService(detector, pipeline)
    path1 = service.get_upload_path("test.jpg")
    path2 = service.get_upload_path("test.jpg")
    assert path1 != path2  # Different UUIDs
    assert str(path1.name).endswith("_test.jpg")
    assert str(path2.name).endswith("_test.jpg")


def test_get_batch_result_not_found():
    """get_batch_result returns None for unknown task."""
    detector = MagicMock()
    pipeline = MagicMock()
    service = DetectionService(detector, pipeline)
    assert service.get_batch_result("nonexistent") is None
