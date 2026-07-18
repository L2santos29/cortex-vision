"""Performance benchmarks for critical paths."""

import numpy as np
import pytest


class TestSanitizeFilenameBench:

    @pytest.mark.benchmark
    def test_sanitize_normal_filename(self, benchmark):
        from src.services import sanitize_filename
        result = benchmark(sanitize_filename, "normal_image_file_name.jpg")
        assert result == "normal_image_file_name.jpg"

    @pytest.mark.benchmark
    def test_sanitize_special_chars(self, benchmark):
        from src.services import sanitize_filename
        result = benchmark(sanitize_filename, "../../etc/passwd$%^&.txt")
        assert result is not None

    @pytest.mark.benchmark
    def test_sanitize_long_filename(self, benchmark):
        from src.services import sanitize_filename
        long_name = "a" * 200 + ".txt"
        result = benchmark(sanitize_filename, long_name)
        assert len(result) <= 128


class TestFrameToBase64Bench:

    @pytest.mark.benchmark
    def test_frame_to_base64_small(self, benchmark):
        from src.utils import frame_to_base64
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = benchmark(frame_to_base64, frame)
        assert result.startswith("data:image/jpeg;base64,")

    @pytest.mark.benchmark
    def test_frame_to_base64_medium(self, benchmark):
        from src.utils import frame_to_base64
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        result = benchmark(frame_to_base64, frame, 320)
        assert result.startswith("data:image/jpeg;base64,")


class TestDrawBoxesBench:

    @pytest.mark.benchmark
    def test_draw_boxes_10_detections(self, benchmark):
        from src.utils import draw_boxes_on_frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"class": f"obj{i}", "confidence": 0.5 + i * 0.05, "bbox": [i * 10, i * 10, i * 10 + 50, i * 10 + 50]}
            for i in range(10)
        ]
        result = benchmark(draw_boxes_on_frame, frame, detections)
        assert result.shape == (480, 640, 3)

    @pytest.mark.benchmark
    def test_draw_boxes_no_detections(self, benchmark):
        from src.utils import draw_boxes_on_frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = benchmark(draw_boxes_on_frame, frame, [])
        assert result.shape == (480, 640, 3)


class TestValidationBench:

    @pytest.mark.benchmark
    def test_validate_image_content_jpeg(self, benchmark):
        from src.utils import validate_image_content
        data = b"\xff\xd8\xff\xe0" + b"x" * 1000
        benchmark(validate_image_content, data)

    @pytest.mark.benchmark
    def test_validate_image_content_png(self, benchmark):
        from src.utils import validate_image_content
        data = b"\x89PNG\r\n\x1a\n" + b"x" * 1000
        benchmark(validate_image_content, data)

    @pytest.mark.benchmark
    def test_validate_image(self, benchmark):
        from src.utils import validate_image
        benchmark(validate_image, "photo.jpg", 1024)
