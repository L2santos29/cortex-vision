"""Unit tests for services.py — DetectionService, CircuitBreaker, DetectionCache, retry."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.services import DetectionService, sanitize_filename
from src.resilience import CircuitBreaker, retry_with_backoff
from src.cache import DetectionCache


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------

class TestSanitizeFilename:

    def test_removes_directory_path(self):
        assert sanitize_filename("/etc/passwd") == "passwd"
        assert sanitize_filename("../../etc/passwd") == "passwd"
        assert sanitize_filename("subdir/file.txt") == "file.txt"

    def test_replaces_special_chars(self):
        result = sanitize_filename("hello$%^world.txt")
        assert "$" not in result
        assert "%" not in result
        assert "^" not in result
        assert result == "hello___world.txt"

    def test_limits_length(self):
        long_name = "a" * 200 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 128

    def test_handles_empty_string(self):
        assert sanitize_filename("") == ""

    def test_handles_no_extension(self):
        result = sanitize_filename("simple_name")
        assert result == "simple_name"

    def test_preserves_unicode(self):
        result = sanitize_filename("foto_ñ.jpg")
        assert result == "foto_ñ.jpg"

    def test_removes_null_bytes(self):
        result = sanitize_filename("file\x00.jpg")
        assert "\x00" not in result

    def test_handles_only_special_chars(self):
        result = sanitize_filename("***.txt")
        assert result == "___.txt"


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:

    def test_closed_by_default(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        assert cb.state == "closed"

    def test_opens_after_failure_threshold(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30.0)
        fn = MagicMock(side_effect=ValueError("boom"))
        with pytest.raises(ValueError):
            cb.call(fn)
        assert cb.failure_count == 1
        assert cb.state == "closed"
        with pytest.raises(ValueError):
            cb.call(fn)
        assert cb.failure_count == 2
        assert cb.state == "open"

    def test_open_raises_without_calling_function(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999.0)
        fn = MagicMock(side_effect=ValueError("boom"))
        with pytest.raises(ValueError):
            cb.call(fn)
        assert cb.state == "open"
        fn2 = MagicMock(return_value="ok")
        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            cb.call(fn2)
        fn2.assert_not_called()

    def test_transitions_to_half_open_after_timeout(self, monkeypatch):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        fn = MagicMock(side_effect=ValueError("boom"))
        with pytest.raises(ValueError):
            cb.call(fn)
        assert cb.state == "open"
        import time
        time.sleep(0.02)
        fn_ok = MagicMock(return_value="recovered")
        result = cb.call(fn_ok)
        assert result == "recovered"
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        fn = MagicMock(return_value="ok")
        result = cb.call(fn)
        assert result == "ok"
        assert cb.failure_count == 0
        assert cb.state == "closed"

    def test_async_call_works(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30.0)

        async def failing():
            raise ValueError("async boom")

        async def ok_func():
            return "async ok"

        async def run():
            with pytest.raises(ValueError):
                await cb.async_call(failing)
            assert cb.state == "closed"
            with pytest.raises(ValueError):
                await cb.async_call(failing)
            assert cb.state == "open"
            with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
                await cb.async_call(ok_func)

        asyncio.run(run())

    def test_async_recovers_after_timeout(self, monkeypatch):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        async def failing():
            raise ValueError("boom")

        async def ok_func():
            return "recovered"

        async def run():
            with pytest.raises(ValueError):
                await cb.async_call(failing)
            assert cb.state == "open"
            import asyncio
            await asyncio.sleep(0.02)
            result = await cb.async_call(ok_func)
            assert result == "recovered"
            assert cb.state == "closed"

        asyncio.run(run())


# ---------------------------------------------------------------------------
# DetectionCache
# ---------------------------------------------------------------------------

class TestDetectionCache:

    def test_get_missing_returns_none(self):
        cache = DetectionCache(max_size=5)
        assert cache.get("missing") is None

    def test_put_and_get(self):
        cache = DetectionCache(max_size=5)
        cache.put("key1", [{"class": "person"}])
        result = cache.get("key1")
        assert result == [{"class": "person"}]

    def test_get_moves_to_end(self):
        cache = DetectionCache(max_size=3)
        cache.put("a", [1])
        cache.put("b", [2])
        cache.put("c", [3])
        cache.get("a")  # Access 'a' — should move to end
        cache.put("d", [4])  # Should evict 'b' (oldest)
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("d") is not None

    def test_evicts_oldest_when_full(self):
        cache = DetectionCache(max_size=2)
        cache.put("a", [1])
        cache.put("b", [2])
        cache.put("c", [3])  # Evicts 'a'
        assert cache.get("a") is None
        assert cache.get("b") == [2]
        assert cache.get("c") == [3]

    def test_clear_empties_cache(self):
        cache = DetectionCache(max_size=5)
        cache.put("a", [1])
        cache.put("b", [2])
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_put_updates_existing(self):
        cache = DetectionCache(max_size=5)
        cache.put("k", [1])
        cache.put("k", [2, 3])
        assert cache.get("k") == [2, 3]

    def test_array_key_uses_hash(self):
        import numpy as np
        cache = DetectionCache(max_size=5)
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.ones((10, 10, 3), dtype=np.uint8)
        k1 = cache._make_array_key(img1)
        k2 = cache._make_array_key(img2)
        assert k1 != k2
        assert k1.startswith("arr:")

    def test_path_key_format(self):
        cache = DetectionCache(max_size=5)
        k = cache._make_key("/tmp/test.jpg")
        assert k == "path:/tmp/test.jpg"


# ---------------------------------------------------------------------------
# retry_with_backoff
# ---------------------------------------------------------------------------

class TestRetryWithBackoff:

    @pytest.mark.asyncio
    async def test_retry_succeeds_eventually(self):
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "success"

        result = await retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        async def always_fails():
            raise OSError("permanent failure")

        with pytest.raises(OSError, match="permanent failure"):
            await retry_with_backoff(always_fails, max_retries=2, base_delay=0.01)

    @pytest.mark.asyncio
    async def test_retry_does_not_retry_non_retryable(self):
        call_count = 0

        async def non_retryable():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await retry_with_backoff(non_retryable, max_retries=2, base_delay=0.01)
        assert call_count == 1  # Should not retry ValueError

    @pytest.mark.asyncio
    async def test_retry_succeeds_first_try(self):
        async def works():
            return "immediate"

        result = await retry_with_backoff(works, max_retries=3, base_delay=0.01)
        assert result == "immediate"


# ---------------------------------------------------------------------------
# DetectionService
# ---------------------------------------------------------------------------

class TestDetectionService:

    def test_init_creates_directories(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.services.Path.exists", lambda self: True)
        from pathlib import Path
        detector = MagicMock()
        pipeline = MagicMock()
        upload_dir = tmp_path / "uploads"
        output_dir = tmp_path / "output"
        service = DetectionService(detector, pipeline, str(upload_dir), str(output_dir))
        assert upload_dir.exists()
        assert output_dir.exists()

    def test_get_upload_path_unique(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        path1 = service.get_upload_path("test.jpg")
        path2 = service.get_upload_path("test.jpg")
        assert path1 != path2
        assert str(path1.name).endswith("_test.jpg")
        assert str(path2.name).endswith("_test.jpg")

    def test_get_batch_upload_path(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        path = service.get_batch_upload_path("task123", "photo.png")
        assert "task123" in str(path.name)
        assert str(path.name).endswith("_photo.png")

    def test_get_batch_result_not_found(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        assert service.get_batch_result("nonexistent") is None

    def test_get_batch_result_found(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        service.batch_results["task1"] = {"detections": [{"class": "person"}]}
        result = service.get_batch_result("task1")
        assert result == {"detections": [{"class": "person"}]}

    def test_validate_image_delegates(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            service.validate_image("bad.gif", 100)

    def test_validate_image_with_content(self, sample_image_bytes):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        # Should pass — JPEG magic bytes
        service.validate_image("photo.jpg", len(sample_image_bytes), sample_image_bytes)

    def test_validate_video_format_valid(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        assert service.validate_video_format("video.mp4") is True
        assert service.validate_video_format("video.avi") is True
        assert service.validate_video_format("video.mov") is True

    def test_validate_video_format_invalid(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        assert service.validate_video_format("video.gif") is False
        assert service.validate_video_format("video.exe") is False

    def test_validate_video_size_within(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        assert service.validate_video_size(100) is True
        assert service.validate_video_size(500 * 1024 * 1024) is True

    def test_validate_video_size_exceeded(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        assert service.validate_video_size(500 * 1024 * 1024 + 1) is False

    def test_export_batch_csv_not_found(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        assert service.export_batch_csv("nonexistent") is None

    def test_export_batch_csv_creates_file(self, tmp_path):
        from pathlib import Path
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline, output_dir=str(tmp_path / "output"))
        service.batch_results["task1"] = {
            "detections": [
                {"image": "img.jpg", "class": "person", "confidence": 0.95, "bbox": [1, 2, 3, 4]},
            ],
        }
        csv_path = service.export_batch_csv("task1")
        assert csv_path is not None
        csv_content = Path(csv_path).read_text()
        assert "image,class,confidence,bbox" in csv_content
        assert "person" in csv_content

    def test_run_detection_uses_cache(self, mock_yolo):
        from src.detector import Detector
        from src.pipeline import BatchPipeline
        import asyncio

        detector = Detector()
        pipeline = BatchPipeline(detector)
        service = DetectionService(detector, pipeline)

        # First call should miss cache, second should hit
        with patch("pathlib.Path.exists", return_value=True):
            mock_yolo.return_value = []
            async def run():
                r1 = await service.run_detection("test.jpg")
                r2 = await service.run_detection("test.jpg")
                return r1, r2
            r1, r2 = asyncio.run(run())
            assert r1 == []
            assert r2 == []

    def test_detection_circuit_protection(self):
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)
        assert service.detection_circuit is not None
        assert service.detection_circuit.failure_threshold == 5

    def test_run_detection_on_array(self, mock_yolo, sample_np_image):
        from src.detector import Detector
        from src.pipeline import BatchPipeline
        import asyncio

        detector = Detector()
        pipeline = BatchPipeline(detector)
        service = DetectionService(detector, pipeline)
        mock_yolo.return_value = []

        async def run():
            return await service.run_detection_on_array(sample_np_image)

        result = asyncio.run(run())
        assert result == []

    def test_run_detection_on_paths(self, mock_yolo):
        from src.detector import Detector
        from src.pipeline import BatchPipeline
        import asyncio
        from tests.conftest import MockResults

        detector = Detector()
        pipeline = BatchPipeline(detector)
        service = DetectionService(detector, pipeline)

        mock_yolo.side_effect = [
            [MockResults(names={0: "person"}, boxes_data=[(0, 0.95, [0, 0, 10, 10])])],
            [MockResults(names={2: "car"}, boxes_data=[(2, 0.85, [5, 5, 15, 15])])],
        ]

        async def run():
            return await service.run_detection_on_paths(["img1.jpg", "img2.jpg"])

        with patch("pathlib.Path.exists", return_value=True):
            result = asyncio.run(run())
        assert len(result) == 2
        assert result[0]["object_count"] == 1

    def test_process_video_saves_and_cleans_up(self, mock_yolo, sample_video_bytes, monkeypatch):
        import asyncio
        from src.detector import Detector
        from src.pipeline import BatchPipeline

        detector = Detector()
        pipeline = BatchPipeline(detector)
        service = DetectionService(detector, pipeline)

        # Mock process_video_frames to return empty
        monkeypatch.setattr("src.services.process_video_frames", lambda *a, **kw: [])

        mock_yolo.return_value = []

        async def run():
            result = await service.process_video(sample_video_bytes, "test.mp4", "vid123")
            return result

        result = asyncio.run(run())
        assert result["filename"] == "test.mp4"
        assert result["video_id"] == "vid123"
        assert result["frames"] == []

    def test_process_batch_stores_results(self, mock_yolo):
        import asyncio
        from src.detector import Detector
        from src.pipeline import BatchPipeline

        detector = Detector()
        pipeline = BatchPipeline(detector)
        service = DetectionService(detector, pipeline)

        mock_yolo.side_effect = [
            RuntimeError("YOLO crash"),  # Make detection fail
        ]

        with patch("pathlib.Path.exists", return_value=True):
            async def run():
                await service.process_batch("batch1", ["img.jpg"])

            asyncio.run(run())
        result = service.get_batch_result("batch1")
        assert result is not None
        assert result["files_processed"] == 1

    def test_process_batch_handles_complete_failure(self):
        import asyncio
        detector = MagicMock()
        pipeline = MagicMock()
        service = DetectionService(detector, pipeline)

        async def run():
            await service.process_batch("fail_batch", ["bad.jpg"])

        asyncio.run(run())
        result = service.get_batch_result("fail_batch")
        assert result is not None
        assert result["files_processed"] == 1
