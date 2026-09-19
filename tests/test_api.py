import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_dummy_image_bytes(format="JPEG") -> io.BytesIO:
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(0, 255, 0))
    img.save(buf, format=format)
    buf.seek(0)
    return buf


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data


def test_health_live_endpoint():
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_ready_endpoint():
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_predict_endpoint_valid_image():
    image_buf = create_dummy_image_bytes(format="JPEG")
    files = {"file": ("test.jpg", image_buf, "image/jpeg")}
    response = client.post("/predict?top_k=2", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.jpg"
    assert "top_prediction" in data
    assert "top_confidence" in data
    assert len(data["predictions"]) <= 2


def test_predict_endpoint_invalid_file_type():
    files = {"file": ("test.txt", io.BytesIO(b"dummy text"), "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
