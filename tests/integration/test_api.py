"""Integration tests — API endpoints + service layer + middleware."""
import pytest


class TestHealthEndpoint:
    """Full health endpoint integration test."""

    def test_health_returns_status_ok(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_returns_model_name(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/health")
        data = resp.json()
        assert "model" in data
        assert data["model"] == "yolov8n.pt"

    def test_health_has_correct_schema(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/health")
        data = resp.json()
        assert set(data.keys()) == {"status", "model"}


class TestAuthIntegration:
    """Auth middleware integrated with endpoints."""

    def test_upload_no_auth_returns_401(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post("/v1/upload", files={"file": ("test.jpg", b"data", "image/jpeg")})
        assert resp.status_code == 401

    def test_upload_with_auth_returns_200_or_400(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("test.jpg", b"data", "image/jpeg")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code in (201, 400)

    def test_batch_upload_no_auth_returns_401(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post("/v1/upload/batch", files={"files": ("test.jpg", b"data", "image/jpeg")})
        assert resp.status_code == 401

    def test_results_no_auth_returns_401(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/v1/results/fake")
        assert resp.status_code == 401

    def test_export_no_auth_returns_401(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/v1/export/fake")
        assert resp.status_code == 401

    def test_detect_frame_no_auth_returns_401(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post("/v1/detect/frame", files={"file": ("frame.jpg", b"data", "image/jpeg")})
        assert resp.status_code == 401

    def test_results_not_found_with_auth(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get(
                "/v1/results/nonexistent",
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 404

    def test_export_not_found_with_auth(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get(
                "/v1/export/nonexistent",
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 404


class TestSecurityHeadersIntegration:

    def test_security_headers_present(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert resp.headers.get("strict-transport-security") is not None
        assert resp.headers.get("x-request-id") is not None

    def test_cors_headers_present(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/health", headers={"Origin": "http://localhost:8000"})
        assert resp.headers.get("access-control-allow-origin") is not None


class TestIndexEndpointIntegration:

    def test_index_returns_200(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/")
        assert resp.status_code == 200
        assert len(resp.text) > 0

    def test_index_sets_api_key_cookie(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/")
        # Should have set-cookie for api_key
        assert "api_key" in resp.headers.get("set-cookie", "").lower()


class TestUploadInvalidInputIntegration:

    def test_upload_invalid_extension_returns_400(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("test.gif", b"data", "image/gif")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 400

    def test_upload_video_invalid_format_returns_400(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload/video",
                files={"file": ("test.gif", b"data", "image/gif")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 400

    def test_detect_frame_empty_returns_400(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/detect/frame",
                files={"file": ("frame.jpg", b"", "image/jpeg")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 400
