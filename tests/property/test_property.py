"""Property-based tests using Hypothesis — finds bugs via random input generation."""

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from src.resilience import CircuitBreaker
from src.cache import DetectionCache
from src.services import sanitize_filename
from src.utils import draw_boxes_on_frame, frame_to_base64


class TestSanitizeFilenameProperties:

    @settings(max_examples=50)
    @given(st.text())
    def test_no_path_separators_in_result(self, text):
        result = sanitize_filename(text)
        assert "/" not in result
        assert "\\" not in result

    @settings(max_examples=50)
    @given(st.text(max_size=100))
    def test_result_length_limited(self, text):
        result = sanitize_filename(text)
        assert len(result) <= 128

    @settings(max_examples=50)
    @given(st.text())
    def test_deterministic(self, text):
        result1 = sanitize_filename(text)
        result2 = sanitize_filename(text)
        assert result1 == result2

    @settings(max_examples=50)
    @given(st.text())
    def test_no_null_bytes(self, text):
        result = sanitize_filename(text)
        assert "\x00" not in result


class TestDrawBoxesProperties:

    @settings(max_examples=20)
    @given(
        st.integers(min_value=10, max_value=200),
        st.integers(min_value=10, max_value=200),
    )
    def test_frame_shape_preserved(self, h, w):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        detections = [{"class": "test", "confidence": 0.5, "bbox": [0, 0, w // 2, h // 2]}]
        result = draw_boxes_on_frame(frame, detections)
        assert result.shape == (h, w, 3)

    @settings(max_examples=20)
    @given(st.lists(
        st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=10
    ))
    def test_any_confidence_level(self, confidences):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{"class": f"obj{i}", "confidence": c, "bbox": [0, 0, 10, 10]}
                      for i, c in enumerate(confidences)]
        result = draw_boxes_on_frame(frame, detections)
        assert result.shape == (100, 100, 3)


class TestFrameToBase64Properties:

    @settings(max_examples=10)
    @given(
        st.integers(min_value=1, max_value=200),
        st.integers(min_value=1, max_value=200),
    )
    def test_always_returns_data_url(self, h, w):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        result = frame_to_base64(frame)
        assert result.startswith("data:image/jpeg;base64,")

    @settings(max_examples=10)
    @given(
        st.integers(min_value=1, max_value=200),
        st.integers(min_value=1, max_value=200),
    )
    def test_url_has_base64_content(self, h, w):
        frame = np.full((h, w, 3), 128, dtype=np.uint8)
        result = frame_to_base64(frame)
        # After "base64," there should be valid base64 data
        b64_part = result.split("base64,")[1]
        assert len(b64_part) > 0
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in b64_part)


class TestCircuitBreakerProperties:

    @settings(max_examples=20)
    @given(
        st.integers(min_value=1, max_value=10),
        st.floats(min_value=0.1, max_value=10.0),
    )
    def test_failure_threshold_behavior(self, threshold, timeout):
        cb = CircuitBreaker(failure_threshold=threshold, recovery_timeout=timeout)
        assert cb.state == "closed"
        assert cb.failure_threshold == threshold
        assert cb.recovery_timeout == timeout


class TestDetectionCacheProperties:

    @settings(max_examples=20)
    @given(st.integers(min_value=1, max_value=20))
    def test_cache_size_respected(self, max_size):
        cache = DetectionCache(max_size=max_size)
        for i in range(max_size * 2):
            cache.put(f"key{i}", [i])
        assert len(cache._cache) <= max_size

    @settings(max_examples=20)
    @given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=20))
    def test_put_get_roundtrip(self, keys):
        cache = DetectionCache(max_size=100)
        for k in keys:
            cache.put(str(k), [k])
        for k in keys:
            val = cache.get(str(k))
            assert val is not None
            assert val == [k]
