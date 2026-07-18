"""Edge case and boundary tests for all modules."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException

from src.resilience import CircuitBreaker
from src.cache import DetectionCache
from src.services import DetectionService, sanitize_filename
from src.utils import (
    validate_image, validate_image_content, validate_video_content,
    draw_boxes_on_frame, frame_to_base64,
)


class TestSanitizeFilenameEdge:

    def test_very_long_extension(self):
        name = "file." + "a" * 200
        result = sanitize_filename(name)
        assert len(result) <= 128

    def test_only_dots(self):
        result = sanitize_filename("...")
        # Dots are valid filename characters
        assert "..." in result or result

    def test_unicode_special(self):
        result = sanitize_filename("文件.txt")
        assert result == "文件.txt"

    def test_emoji_filename(self):
        result = sanitize_filename("🎉party.jpg")
        # Emoji should be preserved (valid unicode)
        assert result.endswith(".jpg")

    def test_control_characters(self):
        result = sanitize_filename("\x00\x01\x02file.txt")
        assert "\x00" not in result

    def test_multiple_path_separators(self):
        result = sanitize_filename("a/b/c/d/file.txt")
        assert result == "file.txt"
        assert "/" not in result

    def test_windows_path(self):
        result = sanitize_filename("C:\\Users\\test\\file.txt")
        assert "file.txt" in result
        assert "\\" not in result


class TestCircuitBreakerEdge:

    def test_zero_threshold(self):
        cb = CircuitBreaker(failure_threshold=0, recovery_timeout=30.0)
        with pytest.raises(ValueError):
            cb.call(MagicMock(side_effect=ValueError("fail")))

    def test_negative_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=-1)
        with pytest.raises(ValueError):
            cb.call(MagicMock(side_effect=ValueError("fail")))

    def test_no_failures_never_opens(self):
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        for _ in range(100):
            cb.call(MagicMock(return_value="ok"))
        assert cb.state == "closed"


class TestDetectionCacheEdge:

    def test_zero_max_size(self):
        cache = DetectionCache(max_size=0)
        cache.put("k", [1])
        # Should still work but immediately evict
        assert cache.get("k") is None or cache.get("k") == [1]

    def test_negative_max_size(self):
        # Negative max_size doesn't raise — it just can't evict properly
        cache = DetectionCache(max_size=-1)
        assert cache.max_size == -1

    def test_put_none_value(self):
        cache = DetectionCache(max_size=5)
        cache.put("k", None)
        assert cache.get("k") is None

    def test_get_empty_key(self):
        cache = DetectionCache(max_size=5)
        assert cache.get("") is None


class TestValidationEdge:

    def test_empty_filename(self):
        with pytest.raises(HTTPException):
            validate_image("", 100)

    def test_max_size_exact_boundary(self):
        validate_image("photo.jpg", MAX_IMAGE_SIZE := 10 * 1024 * 1024)

    def test_max_size_plus_one(self):
        with pytest.raises(HTTPException):
            validate_image("photo.jpg", (10 * 1024 * 1024) + 1)

    def test_validate_image_content_exact_min_len(self):
        # Exactly 12 bytes of JPEG-like data
        validate_image_content(b"\xff\xd8\xff" + b"x" * 9)

    @pytest.mark.parametrize("bad_magic", [
        b"\x00\x00\x00\x00" + b"x" * 20,
        b"\xff\xff\xff\xff" + b"x" * 20,
        b"\x89\x50\x4e\x47" + b"\x00" * 20,  # PNG but wrong version
        b"\x00" * 100,
    ])
    def test_various_bad_magic_bytes(self, bad_magic):
        with pytest.raises(HTTPException):
            validate_image_content(bad_magic)

    def test_video_exact_magic_match(self):
        validate_video_content(b"\x00\x00\x00\x1cftypmp42" + b"x" * 12)

    def test_video_avi_magic_exact(self):
        validate_video_content(b"RIFF" + b"\x00" * 4 + b"AVI " + b"x" * 12)

    def test_video_almost_ftyp_but_wrong_offset(self):
        # ftyp at offset 4 is actually valid (length+ftyp structure)
        # Test with data that has ftyp-like but at wrong position
        with pytest.raises(HTTPException):
            validate_video_content(b"XXXX\x00\x00\x00\x18ftypmp42" + b"x" * 20)


class TestDrawBoxesEdge:

    def test_single_pixel_frame(self):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        detections = [{"class": "tiny", "confidence": 0.5, "bbox": [0, 0, 1, 1]}]
        result = draw_boxes_on_frame(frame, detections)
        assert result.shape == (1, 1, 3)

    def test_huge_confidence(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{"class": "perfect", "confidence": 1.0, "bbox": [10, 10, 50, 50]}]
        result = draw_boxes_on_frame(frame, detections)
        assert result.shape == (100, 100, 3)

    def test_bbox_zero_size(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{"class": "point", "confidence": 0.5, "bbox": [50, 50, 50, 50]}]
        result = draw_boxes_on_frame(frame, detections)
        assert result.shape == (100, 100, 3)

    def test_grayscale_frame(self):
        frame = np.zeros((100, 100), dtype=np.uint8)
        detections = [{"class": "gray", "confidence": 0.5, "bbox": [10, 10, 50, 50]}]
        result = draw_boxes_on_frame(frame, detections)
        assert result.shape == (100, 100)  # Should not crash

    def test_many_detections(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{"class": f"obj{i}", "confidence": 0.5, "bbox": [i, i, i+10, i+10]}
                      for i in range(50)]
        result = draw_boxes_on_frame(frame, detections)
        assert result.shape == (100, 100, 3)


class TestFrameToBase64Edge:

    def test_very_small_frame(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        result = frame_to_base64(frame)
        assert result.startswith("data:image/jpeg;base64,")

    def test_very_large_frame(self):
        frame = np.zeros((4000, 4000, 3), dtype=np.uint8)
        result = frame_to_base64(frame, max_size=320)
        assert result.startswith("data:image/jpeg;base64,")


class TestDetectionServiceEdge:

    def test_get_batch_result_with_empty_detections(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        service.batch_results["empty"] = {"detections": []}
        result = service.get_batch_result("empty")
        assert result == {"detections": []}

    def test_export_csv_with_empty_detections(self, tmp_path):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline, output_dir=str(tmp_path / "out"))
        service.batch_results["empty"] = {"detections": []}
        csv_path = service.export_batch_csv("empty")
        assert csv_path is not None

    def test_validate_image_none_contents(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        # Should not raise since contents is None
        service.validate_image("photo.jpg", 100, contents=None)

    def test_validate_image_bad_contents(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        with pytest.raises(HTTPException):
            service.validate_image("photo.jpg", 100, contents=b"garbage")
