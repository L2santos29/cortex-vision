"""Integration tests for the FastAPI application (API contract + integration)."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a TestClient with YOLO mocked to prevent actual model load."""
    with patch("ultralytics.YOLO") as mock_yolo_class:
        mock_yolo_instance = MagicMock()
        mock_yolo_class.return_value = mock_yolo_instance
        mock_yolo_instance.return_value = []

        from src.main import app

        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    """GET /health returns 200 with status ok and model name."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert data["model"] is not None


def test_health_endpoint_contract(client):
    """GET /health response matches expected schema."""
    response = client.get("/health")
    data = response.json()
    # Contract: must have exactly these keys
    assert set(data.keys()) == {"status", "model"}
    assert isinstance(data["status"], str)
    assert isinstance(data["model"], str)


def test_index_endpoint(client):
    """GET / returns 200 (HTML or fallback text)."""
    response = client.get("/")
    assert response.status_code == 200
    assert len(response.text) > 0


def test_upload_no_auth(client):
    """POST /v1/upload without API key returns 401."""
    response = client.post("/v1/upload", files={"file": ("test.jpg", b"fake-image-data", "image/jpeg")})
    assert response.status_code == 401
    data = response.json()
    assert "detail" in data


def test_upload_with_auth(client):
    """POST /v1/upload with valid API key returns 201 or appropriate error."""
    response = client.post(
        "/v1/upload",
        files={"file": ("test.jpg", b"fake-image-data", "image/jpeg")},
        headers={"X-API-Key": "test-key-for-testing"},
    )
    # Can be 201 (processed) or 400 (validation error) — but not 401
    assert response.status_code in (201, 400)
    assert response.status_code != 401


def test_upload_invalid_file_type(client):
    """POST /v1/upload with invalid extension returns 400."""
    response = client.post(
        "/v1/upload",
        files={"file": ("test.gif", b"fake-gif-data", "image/gif")},
        headers={"X-API-Key": "test-key-for-testing"},
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data or "error" in data


def test_upload_video_no_auth(client):
    """POST /v1/upload/video without API key returns 401."""
    response = client.post(
        "/v1/upload/video",
        files={"file": ("test.mp4", b"fake-video-data", "video/mp4")},
    )
    assert response.status_code == 401


def test_upload_batch_no_auth(client):
    """POST /v1/upload/batch without API key returns 401."""
    response = client.post(
        "/v1/upload/batch",
        files={"files": ("test.jpg", b"data", "image/jpeg")},
    )
    assert response.status_code == 401


def test_detect_frame_with_auth(client):
    """POST /v1/detect/frame with valid auth processes frame."""
    response = client.post(
        "/v1/detect/frame",
        files={"file": ("frame.jpg", b"fake-jpeg-data", "image/jpeg")},
        headers={"X-API-Key": "test-key-for-testing"},
    )
    # Either processed (200) or invalid image data (400)
    assert response.status_code in (200, 400)


def test_results_endpoint_no_auth(client):
    """GET /v1/results/{task_id} without API key returns 401."""
    response = client.get("/v1/results/fake-task-id")
    assert response.status_code == 401


def test_results_endpoint_not_found(client):
    """GET /v1/results/{task_id} with auth but invalid ID returns 404."""
    response = client.get(
        "/v1/results/nonexistent-task",
        headers={"X-API-Key": "test-key-for-testing"},
    )
    assert response.status_code == 404


def test_export_csv_no_auth(client):
    """GET /v1/export/{task_id} without API key returns 401."""
    response = client.get("/v1/export/fake-task-id")
    assert response.status_code == 401


def test_export_csv_not_found(client):
    """GET /v1/export/{task_id} with auth but invalid ID returns 404."""
    response = client.get(
        "/v1/export/nonexistent-task",
        headers={"X-API-Key": "test-key-for-testing"},
    )
    assert response.status_code == 404


def test_security_headers_present(client):
    """Response includes security headers from middleware."""
    response = client.get("/health")
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("strict-transport-security") is not None
    assert response.headers.get("x-request-id") is not None


def test_cors_headers_present(client):
    """CORS headers are present in responses."""
    response = client.get("/health", headers={"Origin": "http://localhost:8000"})
    assert response.headers.get("access-control-allow-origin") is not None
