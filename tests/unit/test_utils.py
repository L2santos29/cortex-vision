"""Unit tests for utils.py — validation, frame processing, image utilities."""

import numpy as np
import pytest
from fastapi import HTTPException

from src.utils import (
    ALLOWED_EXTENSIONS,
    MAX_IMAGE_SIZE,
    MAX_VIDEO_SIZE,
    ALLOWED_VIDEO_EXTENSIONS,
    validate_image,
    validate_image_content,
    validate_video_content,
    extract_frames,
    draw_boxes_on_frame,
    frame_to_base64,
    process_video_frames,
)


# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------

class TestValidateImage:

    def test_valid_extensions_pass(self):
        for ext in ALLOWED_EXTENSIONS:
            validate_image(f"photo{ext}", 1024)
        validate_image("photo.jpg", MAX_IMAGE_SIZE)

    def test_invalid_extension_raises(self):
        with pytest.raises(HTTPException) as exc:
            validate_image("photo.gif", 1024)
        assert exc.value.status_code == 400

        with pytest.raises(HTTPException) as exc:
            validate_image("file.pdf", 1024)
        assert exc.value.status_code == 400

    def test_oversized_file_raises(self):
        with pytest.raises(HTTPException) as exc:
            validate_image("photo.jpg", MAX_IMAGE_SIZE + 1)
        assert exc.value.status_code == 400
        assert "too large" in exc.value.detail.lower()

    def test_zero_size_passes(self):
        validate_image("empty.jpg", 0)

    def test_no_extension(self):
        with pytest.raises(HTTPException):
            validate_image("noext", 100)

    def test_uppercase_extension(self):
        validate_image("photo.JPG", 100)
        validate_image("photo.PNG", 100)


# ---------------------------------------------------------------------------
# validate_image_content (magic bytes)
# ---------------------------------------------------------------------------

class TestValidateImageContent:

    def test_jpeg_magic_bytes(self):
        validate_image_content(b"\xff\xd8\xff\xe0" + b"x" * 20)

    def test_png_magic_bytes(self):
        validate_image_content(b"\x89PNG\r\n\x1a\n" + b"x" * 20)

    def test_bmp_magic_bytes(self):
        validate_image_content(b"BM" + b"x" * 20)

    def test_webp_magic_bytes(self):
        validate_image_content(b"RIFF" + b"\x00" * 4 + b"WEBP")

    def test_invalid_content_raises(self):
        with pytest.raises(HTTPException) as exc:
            validate_image_content(b"garbage data that is not an image")
        assert exc.value.status_code == 400

    def test_too_short_content_raises(self):
        with pytest.raises(HTTPException) as exc:
            validate_image_content(b"short")
        assert exc.value.status_code == 400
        assert "too small" in exc.value.detail.lower()

    def test_wrong_riFF_not_webp_raises(self):
        with pytest.raises(HTTPException):
            validate_image_content(b"RIFF" + b"\x00" * 4 + b"NOTW")


# ---------------------------------------------------------------------------
# validate_video_content
# ---------------------------------------------------------------------------

class TestValidateVideoContent:

    def test_mp4_magic_ftyp(self):
        validate_video_content(b"\x00\x00\x00\x1cftypmp42" + b"x" * 20)

    def test_avi_magic(self):
        validate_video_content(b"RIFF" + b"\x00" * 4 + b"AVI " + b"x" * 10)

    def test_mkv_magic(self):
        validate_video_content(b"\x1a\x45\xdf\xa3" + b"x" * 20)

    def test_invalid_video_raises(self):
        with pytest.raises(HTTPException) as exc:
            validate_video_content(b"not a video file at all")
        assert exc.value.status_code == 400

    def test_too_short_video_raises(self):
        with pytest.raises(HTTPException) as exc:
            validate_video_content(b"short")
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# draw_boxes_on_frame
# ---------------------------------------------------------------------------

class TestDrawBoxesOnFrame:

    def test_returns_modified_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"class": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]},
        ]
        result = draw_boxes_on_frame(frame, detections)
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)

    def test_empty_detections_returns_unchanged(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_boxes_on_frame(frame, [])
        assert np.array_equal(result, frame)

    def test_clamps_out_of_bounds(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {"class": "person", "confidence": 0.95, "bbox": [-50, -50, 500, 500]},
        ]
        result = draw_boxes_on_frame(frame, detections)
        assert result.shape == (100, 100, 3)

    def test_multiple_detections(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"class": "person", "confidence": 0.95, "bbox": [10, 10, 50, 100]},
            {"class": "car", "confidence": 0.5, "bbox": [200, 100, 400, 300]},
            {"class": "dog", "confidence": 0.3, "bbox": [0, 0, 30, 30]},
        ]
        result = draw_boxes_on_frame(frame, detections)
        assert result.shape == (480, 640, 3)

    def test_zero_confidence(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{"class": "bg", "confidence": 0.0, "bbox": [0, 0, 10, 10]}]
        result = draw_boxes_on_frame(frame, detections)
        assert result.shape == (100, 100, 3)


# ---------------------------------------------------------------------------
# frame_to_base64
# ---------------------------------------------------------------------------

class TestFrameToBase64:

    def test_returns_base64_data_url(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = frame_to_base64(frame)
        assert isinstance(result, str)
        assert result.startswith("data:image/jpeg;base64,")

    def test_minimal_frame(self):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        result = frame_to_base64(frame)
        assert isinstance(result, str)
        assert result.startswith("data:image/jpeg;base64,")

    def test_downscales_large_frames(self):
        frame = np.zeros((2000, 2000, 3), dtype=np.uint8)
        result = frame_to_base64(frame, max_size=320)
        assert isinstance(result, str)
        assert result.startswith("data:image/jpeg;base64,")

    def test_all_white_frame(self):
        frame = np.full((50, 50, 3), 255, dtype=np.uint8)
        result = frame_to_base64(frame)
        assert result.startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# extract_frames (mocked cv2)
# ---------------------------------------------------------------------------

class TestExtractFrames:

    def test_raises_on_unopenable_video(self):
        with pytest.raises(RuntimeError, match="Failed to open video"):
            extract_frames("nonexistent.mp4")
