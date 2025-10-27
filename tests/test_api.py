"""Basic API tests."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()
    assert response.json()["name"] == "Knowledge Base API"


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_docs_available():
    """Test that API docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_upload_without_auth():
    """Test that upload requires authentication."""
    response = client.post("/documents/")
    assert response.status_code == 401


def test_ask_without_auth():
    """Test that ask requires authentication."""
    response = client.post("/ask", json={"query": "test"})
    assert response.status_code == 401
