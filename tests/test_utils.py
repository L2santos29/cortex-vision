"""Tests for utility functions."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException

from src.utils import (
    ALLOWED_EXTENSIONS,
    MAX_IMAGE_SIZE,
    draw_boxes_on_frame,
    extract_frames,
    frame_to_base64,
    validate_image,
)


# --- validate_image ---


def test_validate_image_valid():
    """Valid extensions and sizes pass without error."""
    for ext in ALLOWED_EXTENSIONS:
        validate_image(f"photo{ext}", 1024)
    validate_image("photo.jpg", MAX_IMAGE_SIZE)


def test_validate_image_invalid_extension():
    """Unsupported extensions raise HTTPException 400."""
    with pytest.raises(HTTPException) as exc:
        validate_image("photo.gif", 1024)
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        validate_image("document.pdf", 1024)
    assert exc.value.status_code == 400


def test_validate_image_too_large():
    """File exceeding MAX_IMAGE_SIZE raises HTTPException 400."""
    with pytest.raises(HTTPException) as exc:
        validate_image("photo.jpg", MAX_IMAGE_SIZE + 1)
    assert exc.value.status_code == 400
    assert "too large" in exc.value.detail.lower()


# --- sanitize_filename (defined in src.main) ---


def test_sanitize_filename_removes_path():
    """Path components are stripped, leaving only the filename."""
    from src.main import sanitize_filename

    assert sanitize_filename("/etc/passwd") == "passwd"
    assert sanitize_filename("../../etc/passwd") == "passwd"


def test_sanitize_filename_removes_special_chars():
    """Special characters are replaced with underscores."""
    from src.main import sanitize_filename

    result = sanitize_filename("hello$%^world.txt")
    assert "$" not in result
    assert "%" not in result
    assert "^" not in result
    # Special chars become underscores
    assert result == "hello___world.txt"


def test_sanitize_filename_limits_length():
    """Filename is truncated to 128 characters."""
    from src.main import sanitize_filename

    long_name = "a" * 200 + ".txt"
    result = sanitize_filename(long_name)
    assert len(result) <= 128


# --- frame_to_base64 ---


def test_frame_to_base64_returns_string():
    """frame_to_base64 returns a base64 data URL."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = frame_to_base64(frame)
    assert isinstance(result, str)
    assert result.startswith("data:image/jpeg;base64,")


# --- extract_frames ---


@patch("src.utils.cv2.VideoCapture")
def test_extract_frames_invalid_path(mock_vc):
    """extract_frames raises ValueError for unopenable video."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_vc.return_value = mock_cap

    with pytest.raises(RuntimeError, match="Failed to open video"):
        extract_frames("nonexistent.mp4")


# --- draw_boxes_on_frame ---


def test_draw_boxes_on_frame_returns_frame():
    """draw_boxes_on_frame returns a modified frame without crashing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        {"class": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]},
        {"class": "car", "confidence": 0.5, "bbox": [300, 50, 500, 150]},
    ]
    result = draw_boxes_on_frame(frame, detections)
    assert isinstance(result, np.ndarray)
    assert result.shape == (480, 640, 3)
