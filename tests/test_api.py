"""Integration tests for the FastAPI application."""

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

        # Import inside the patch — module-level code runs now but YOLO
        # is never actually called because of lazy loading in Detector.
        from src.main import app

        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    """GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_index_endpoint(client):
    """GET / returns 200 (HTML or fallback text)."""
    response = client.get("/")
    assert response.status_code == 200
    # The response can either be HTML or a simple <h1> fallback
    assert response.text.startswith("<")
