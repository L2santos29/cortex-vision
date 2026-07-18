"""Regression tests — prevent known bugs from recurring."""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestDetectorRegression:
    """Regression tests for previously fixed or known detection bugs."""

    def test_detect_raises_on_nonexistent_file(self):
        """Regression: FileNotFoundError for missing images."""
        from src.detector import Detector
        detector = Detector()
        with pytest.raises(FileNotFoundError, match="not found"):
            detector.detect("/nonexistent/path/image.jpg")

    def test_detect_returns_empty_for_no_boxes(self, mock_yolo, monkeypatch):
        """Regression: Empty image returns [], not None or crash."""
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
        from src.detector import Detector
        from tests.conftest import MockResults

        mock_yolo.return_value = [MockResults(names={0: "person"}, boxes_data=None)]
        detector = Detector()
        detections = detector.detect("empty.jpg")
        assert detections == []

    def test_detect_array_handles_empty_numpy(self, mock_yolo):
        """Regression: Empty numpy array doesn't crash detector."""
        from src.detector import Detector
        from tests.conftest import MockResults

        mock_yolo.return_value = [MockResults(names={0: "person"}, boxes_data=None)]
        detector = Detector()
        empty_img = np.empty((0, 0, 3), dtype=np.uint8)
        detections = detector.detect_array(empty_img)
        assert detections == []

    def test_parse_results_handles_multiple_results(self, mock_yolo):
        """Regression: Multiple YOLO result objects are all parsed."""
        from src.detector import Detector
        from tests.conftest import MockResults

        detector = Detector()
        results = [
            MockResults(names={0: "person"}, boxes_data=[(0, 0.95, [0, 0, 10, 10])]),
            MockResults(names={0: "person"}, boxes_data=[(0, 0.90, [5, 5, 15, 15])]),
        ]
        parsed = detector._parse_results(results)
        assert len(parsed) == 2


class TestPipelineRegression:

    def test_aggregate_empty_list(self):
        """Regression: Empty per_file_results returns empty list, not crash."""
        from src.pipeline import BatchPipeline
        from unittest.mock import MagicMock
        pipeline = BatchPipeline(MagicMock())
        aggregated = pipeline.aggregate([])
        assert aggregated == []

    def test_stats_empty_list(self):
        """Regression: Empty aggregated returns zero counts, not crash."""
        from src.pipeline import BatchPipeline
        from unittest.mock import MagicMock
        pipeline = BatchPipeline(MagicMock())
        stats = pipeline.stats([])
        assert stats["total_detections"] == 0
        assert stats["unique_classes"] == 0


class TestServicesRegression:

    def test_export_csv_empty_detections_no_crash(self, tmp_path):
        """Regression: CSV export with empty detection list doesn't crash."""
        from src.services import DetectionService
        from unittest.mock import MagicMock
        service = DetectionService(MagicMock(), MagicMock(), output_dir=str(tmp_path / "out"))
        service.batch_results["empty"] = {"detections": []}
        csv_path = service.export_batch_csv("empty")
        assert csv_path is not None

    def test_process_batch_with_zero_files(self):
        """Regression: process_batch with empty paths doesn't crash."""
        import asyncio
        from src.services import DetectionService
        from unittest.mock import MagicMock
        service = DetectionService(MagicMock(), MagicMock())
        async def run():
            await service.process_batch("empty_batch", [])
        asyncio.run(run())
        result = service.get_batch_result("empty_batch")
        assert result is not None
        assert result["files_processed"] == 0

    def test_validate_image_checks_content(self):
        """Regression: validate_image with contents checks magic bytes, not just extension."""
        from src.services import DetectionService
        from unittest.mock import MagicMock
        from fastapi import HTTPException
        service = DetectionService(MagicMock(), MagicMock())
        # Valid extension but invalid content → should raise
        with pytest.raises(HTTPException):
            service.validate_image("photo.jpg", 100, contents=b"not an image")


class TestUtilsRegression:

    def test_validate_image_content_rejects_garbage(self):
        """Regression: validate_image_content rejects non-image data."""
        from src.utils import validate_image_content
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            validate_image_content(b"\x00\x01\x02\x03\x04\x05" * 10)

    def test_validate_video_content_rejects_garbage(self):
        """Regression: validate_video_content rejects non-video data."""
        from src.utils import validate_video_content
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            validate_video_content(b"\x00\x01\x02\x03\x04\x05" * 10)

    def test_draw_boxes_clamps_negative_coordinates(self):
        """Regression: Negative bbox coordinates clamped to 0, no crash."""
        from src.utils import draw_boxes_on_frame
        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{"class": "person", "confidence": 0.95, "bbox": [-50, -50, 500, 500]}]
        result = draw_boxes_on_frame(frame, detections)
        assert result.shape == (100, 100, 3)

    def test_frame_to_base64_minimal_frame(self):
        """Regression: Minimal 1x1 frame encodes without error."""
        from src.utils import frame_to_base64
        import numpy as np
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        result = frame_to_base64(frame)
        assert result.startswith("data:image/jpeg;base64,")


class TestCircuitBreakerRegression:

    def test_async_call_recovers(self):
        """Regression: Circuit breaker async transitions properly from open to half-open."""
        from src.resilience import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        async def failing():
            raise ValueError("transient")

        async def ok():
            return "recovered"

        async def run():
            with pytest.raises(ValueError):
                await cb.async_call(failing)
            assert cb.state == "open"
            await asyncio.sleep(0.02)
            result = await cb.async_call(ok)
            assert result == "recovered"
            assert cb.state == "closed"

        asyncio.run(run())
