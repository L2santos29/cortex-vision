"""Smoke tests — verify the system loads and basic operations work."""

import pytest


class TestSmokeImports:
    """All modules import without error."""

    def test_import_config(self):
        from src.config import Settings, settings
        assert settings is not None
        assert hasattr(settings, "api_key")

    def test_import_detector(self):
        from src.detector import Detector
        assert Detector is not None

    def test_import_pipeline(self):
        from src.pipeline import BatchPipeline
        assert BatchPipeline is not None

    def test_import_services(self):
        from src.services import DetectionService, sanitize_filename
        from src.resilience import CircuitBreaker, retry_with_backoff
        from src.cache import DetectionCache
        assert all(x is not None for x in [DetectionService, sanitize_filename, retry_with_backoff, CircuitBreaker, DetectionCache])

    def test_import_utils(self):
        from src.utils import (
            validate_image, validate_image_content, validate_video_content,
            extract_frames, draw_boxes_on_frame, frame_to_base64, process_video_frames,
        )
        assert all(x is not None for x in [validate_image, validate_image_content, validate_video_content,
                                            extract_frames, draw_boxes_on_frame, frame_to_base64, process_video_frames])

    def test_import_middleware(self):
        from src.middleware import (
            verify_api_key, HTTPSRedirectMiddleware, SecurityHeadersMiddleware,
            RateLimitMiddleware, RequestIDMiddleware, MetricsMiddleware, metrics_endpoint,
        )
        assert all(x is not None for x in [verify_api_key, HTTPSRedirectMiddleware, SecurityHeadersMiddleware,
                                            RateLimitMiddleware, RequestIDMiddleware, MetricsMiddleware, metrics_endpoint])

    def test_import_profiling(self):
        from src.profiling import timed, timed_context
        assert timed is not None
        assert timed_context is not None

    def test_import_cli(self):
        from src.cli import create_parser, main
        assert create_parser is not None
        assert main is not None

    def test_import_main_app(self):
        from src.main import app
        assert app is not None
        assert app.title == "Cortex-Vision"


class TestSmokeInstantiation:

    def test_detector_creates(self):
        from src.detector import Detector
        d = Detector()
        assert d.model_name == "yolov8n.pt"
        assert d._model is None  # Lazy loading

    def test_pipeline_creates(self, mock_detector):
        from src.pipeline import BatchPipeline
        p = BatchPipeline(mock_detector)
        assert p.detector is mock_detector

    def test_service_creates(self, mock_detector, mock_pipeline):
        from src.services import DetectionService
        s = DetectionService(mock_detector, mock_pipeline)
        assert s.detector is mock_detector
        assert s.pipeline is mock_pipeline
        assert s.upload_dir.exists()
        assert s.output_dir.exists()

    def test_settings_required_key(self, monkeypatch):
        monkeypatch.delenv("API_KEY", raising=False)
        from src.config import Settings
        import pytest
        with pytest.raises(Exception):
            Settings()

    def test_circuit_breaker_creates(self):
        from src.resilience import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_detection_cache_creates(self):
        from src.cache import DetectionCache
        cache = DetectionCache(max_size=10)
        assert cache.max_size == 10
        assert len(cache._cache) == 0
