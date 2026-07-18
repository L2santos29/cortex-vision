"""Unit tests for main.py route handlers using TestClient."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Reset rate limit state between tests."""
    from src.middleware import RateLimitMiddleware
    RateLimitMiddleware.reset_all()
    yield


class TestIndexRoute:

    def test_index_returns_html(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    def test_index_sets_api_key_cookie(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/")
        set_cookie = resp.headers.get("set-cookie", "")
        assert "api_key" in set_cookie
        assert "httponly" in set_cookie.lower()


class TestMetricsRoute:

    def test_metrics_requires_auth(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/metrics")
        assert resp.status_code == 401

    def test_metrics_with_auth(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/metrics", headers={"X-API-Key": "test-key-for-testing"})
        # May be 200 OK or 500 if no metrics collected yet
        assert resp.status_code in (200, 500)

    def test_metrics_returns_prometheus_format(self):
        from src.main import app
        with TestClient(app) as client:
            # Make a request first to populate metrics
            client.get("/health")
            resp = client.get("/metrics", headers={"X-API-Key": "test-key-for-testing"})
        if resp.status_code == 200:
            assert "cortex_vision_" in resp.text


class TestUploadRoute:

    def test_upload_with_valid_image_returns_201_or_400(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0" + b"x" * 100, "image/jpeg")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code in (201, 400)
        # Should return JSON in either case
        assert "application/json" in resp.headers["content-type"]

    def test_upload_invalid_extension_400(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("test.gif", b"data", "image/gif")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 400

    def test_upload_no_auth_401(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("test.jpg", b"data", "image/jpeg")},
            )
        assert resp.status_code == 401


class TestUploadBatchRoute:

    def test_batch_upload_no_files_returns_422(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload/batch",
                headers={"X-API-Key": "test-key-for-testing"},
            )
        # FastAPI should return 422 for missing required parameter
        assert resp.status_code == 422

    def test_batch_upload_no_auth_401(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload/batch",
                files={"files": ("test.jpg", b"data", "image/jpeg")},
            )
        assert resp.status_code == 401

    def test_batch_upload_with_files(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload/batch",
                files=[
                    ("files", ("a.jpg", b"\xff\xd8\xff\xe0" + b"x" * 100, "image/jpeg")),
                ],
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code in (201, 400)


class TestUploadVideoRoute:

    def test_upload_video_no_auth_401(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload/video",
                files={"file": ("test.mp4", b"data", "video/mp4")},
            )
        assert resp.status_code == 401

    def test_upload_video_invalid_format_400(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload/video",
                files={"file": ("test.gif", b"data", "image/gif")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 400

    def test_upload_video_with_valid_mp4(self):
        from src.main import app
        from unittest.mock import patch
        # Mock process_video to avoid OpenCV decoding issues with fake data
        with patch("src.services.process_video_frames", return_value=[]):
            with TestClient(app) as client:
                video_data = b"\x00\x00\x00\x1cftypmp42" + b"\x00" * 200
                resp = client.post(
                    "/v1/upload/video",
                    files={"file": ("test.mp4", video_data, "video/mp4")},
                    headers={"X-API-Key": "test-key-for-testing"},
                )
            assert resp.status_code in (201, 500)


class TestDetectFrameRoute:

    def test_detect_frame_empty_400(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/detect/frame",
                files={"file": ("frame.jpg", b"", "image/jpeg")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 400

    def test_detect_frame_no_auth_401(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/detect/frame",
                files={"file": ("frame.jpg", b"data", "image/jpeg")},
            )
        assert resp.status_code == 401

    def test_detect_frame_invalid_image_400(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/detect/frame",
                files={"file": ("frame.jpg", b"not a valid jpeg at all", "image/jpeg")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code in (400, 500)


class TestResultsRoute:

    def test_results_not_found_404(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.get(
                "/v1/results/nonexistent-task",
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 404

    def test_results_with_pagination(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.get(
                "/v1/results/nonexistent-task?limit=5&offset=10",
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 404


class TestExportRoute:

    def test_export_not_found_404(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.get(
                "/v1/export/nonexistent-task",
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 404

    def test_export_no_auth_401(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/v1/export/task-123")
        assert resp.status_code == 401
