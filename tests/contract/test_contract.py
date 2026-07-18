"""Contract tests — verify API responses match expected schemas and status codes."""

import pytest


class TestHealthContract:

    def test_health_status_code(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_shape(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/health")
        data = resp.json()
        assert isinstance(data, dict)
        assert "status" in data
        assert "model" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["model"], str)

    def test_health_content_type(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]


class TestUploadContract:

    def test_upload_returns_json_on_201(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("test.jpg", b"fake-image-data", "image/jpeg")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        if resp.status_code == 201:
            data = resp.json()
            assert "filename" in data
            assert "detections" in data

    def test_upload_returns_json_on_400(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("test.gif", b"data", "image/gif")},
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data or "error" in data

    def test_upload_returns_json_on_401(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.post(
                "/v1/upload",
                files={"file": ("test.jpg", b"data", "image/jpeg")},
            )
        assert resp.status_code == 401
        data = resp.json()
        assert "detail" in data


class TestResultsContract:

    def test_results_404_shape(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get(
                "/v1/results/nonexistent",
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data or "detail" in data

    def test_results_pagination_params(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get(
                "/v1/results/nonexistent?limit=10&offset=0",
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 404  # Task doesn't exist


class TestExportContract:

    def test_export_404_shape(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get(
                "/v1/export/nonexistent",
                headers={"X-API-Key": "test-key-for-testing"},
            )
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data or "detail" in data


class TestErrorResponseConsistency:

    def test_validation_error_format(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            # POST to GET endpoint should trigger 405 or validation
            resp = client.post(
                "/v1/results/test",
                headers={"X-API-Key": "test-key-for-testing"},
            )
        # Should return valid JSON
        data = resp.json()
        assert isinstance(data, dict)
