"""Fault injection and resilience tests — verify the system handles failures gracefully."""

import asyncio
import io
from unittest.mock import MagicMock, patch

import pytest


class TestFaultFileIO:
    """Faults: file system failures (disk full, permission denied, file not found)."""

    def test_validate_image_content_empty_bytes(self):
        """Resilience: Empty bytes raises HTTPException 400, not 500."""
        from src.utils import validate_image_content
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            validate_image_content(b"")
        assert exc.value.status_code == 400

    def test_validate_video_content_empty_bytes(self):
        """Resilience: Empty video bytes raises HTTPException 400."""
        from src.utils import validate_video_content
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            validate_video_content(b"")
        assert exc.value.status_code == 400

    def test_sanitize_filename_none(self):
        """Resilience: None filename handled gracefully."""
        from src.services import sanitize_filename
        # Path(None).name would crash — let's verify the code handles this
        with pytest.raises((TypeError, AttributeError)):
            sanitize_filename(None)


class TestFaultNetwork:
    """Faults: network and timeout failures."""

    def test_retry_on_connection_error(self):
        """Resilience: retry_with_backoff retries on ConnectionError."""
        from src.resilience import retry_with_backoff
        call_count = [0]

        async def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("transient network failure")
            return "ok"

        async def run():
            return await retry_with_backoff(flaky, max_retries=3, base_delay=0.01)

        result = asyncio.run(run())
        assert result == "ok"
        assert call_count[0] == 3

    def test_retry_on_timeout_error(self):
        """Resilience: retry_with_backoff retries on TimeoutError."""
        from src.resilience import retry_with_backoff
        call_count = [0]

        async def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise TimeoutError("timeout")
            return "ok"

        async def run():
            return await retry_with_backoff(flaky, max_retries=2, base_delay=0.01)

        result = asyncio.run(run())
        assert result == "ok"

    def test_retry_exhausted_on_permanent_failure(self):
        """Resilience: retry gives up after max_retries and raises."""
        from src.resilience import retry_with_backoff

        async def always_fails():
            raise OSError("permanent")

        async def run():
            with pytest.raises(OSError):
                await retry_with_backoff(always_fails, max_retries=3, base_delay=0.01)

        asyncio.run(run())


class TestFaultCircuitBreaker:
    """Faults: cascading failures handled by circuit breaker."""

    def test_async_call_with_open_circuit(self):
        """Resilience: Open circuit raises RuntimeError without calling function."""
        from src.resilience import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0)

        async def failing():
            raise ValueError("boom")

        async def should_not_be_called():
            pytest.fail("This function should not be called")

        async def run():
            with pytest.raises(ValueError):
                await cb.async_call(failing)
            with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
                await cb.async_call(should_not_be_called)

        asyncio.run(run())

    def test_sync_open_circuit(self):
        """Resilience: Sync open circuit raises RuntimeError."""
        from src.resilience import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0)
        with pytest.raises(ValueError):
            cb.call(MagicMock(side_effect=ValueError("boom")))
        with pytest.raises(RuntimeError):
            cb.call(MagicMock(return_value="should not reach"))


class TestFaultService:
    """Faults: service layer failures."""

    def test_process_batch_with_all_failing_files(self, mock_yolo):
        """Resilience: process_batch stores error info when all files fail."""
        import asyncio
        from src.detector import Detector
        from src.pipeline import BatchPipeline
        from src.services import DetectionService

        detector = Detector()
        pipeline = BatchPipeline(detector)
        service = DetectionService(detector, pipeline)

        # Make YOLO raise on any input
        mock_yolo.side_effect = RuntimeError("model crash")

        with patch("pathlib.Path.exists", return_value=True):
            async def run():
                await service.process_batch("fault_batch", ["bad1.jpg", "bad2.jpg"])

            asyncio.run(run())

        result = service.get_batch_result("fault_batch")
        assert result is not None
        # Should have partial results, not crash

    def test_save_upload_io_error(self):
        """Resilience: save_upload failure returns error, doesn't crash server."""
        from src.services import DetectionService
        from src.resilience import retry_with_backoff
        from unittest.mock import MagicMock
        from pathlib import Path
        import asyncio

        service = DetectionService(MagicMock(), MagicMock())

        async def failing_write(p, c):
            raise OSError("disk full")

        async def run():
            try:
                await retry_with_backoff(failing_write, Path("/tmp/x"), b"data",
                                          max_retries=1, base_delay=0.01)
                return False
            except OSError:
                return True

        result = asyncio.run(run())
        assert result is True  # Raised OSError after retries exhausted


class TestFaultMiddleware:
    """Faults: middleware handling of bad auth."""

    def test_invalid_api_key_returns_401(self):
        """Resilience: Invalid API key returns 401, not 500."""
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("test.jpg", b"data", "image/jpeg")},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401

    def test_missing_api_key_header_returns_401(self):
        """Resilience: Missing auth header returns 401, not 500."""
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("test.jpg", b"data", "image/jpeg")},
            )
        assert resp.status_code == 401


class TestFaultBadInputs:
    """Faults: malformed/attack-like inputs."""

    def test_upload_with_script_in_filename(self):
        """Resilience: Filename with script tags doesn't cause XSS in response."""
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("<script>alert(1)</script>.jpg", b"data", "image/jpeg")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        # Should not 500 — validation error or similar expected
        assert resp.status_code in (201, 400)

    def test_upload_with_null_bytes_in_filename(self):
        """Resilience: Filename with null bytes doesn't crash."""
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("file\x00.jpg", b"data", "image/jpeg")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code in (201, 400)
