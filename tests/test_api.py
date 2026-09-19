from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "image-classification-api"
    assert data["status"] == "running"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_predict_with_invalid_file():

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_predict_with_real_image():
    cat_files = list(Path("data/raw/test/cat").glob("*.jpg"))
    if cat_files:
        sample_cat = cat_files[0]
        with open(sample_cat, "rb") as f:
            img_bytes = f.read()
        filename = sample_cat.name
    else:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), color=(255, 0, 0)).save(buf, format="JPEG")
        img_bytes = buf.getvalue()
        filename = "test_cat.jpg"

    response = client.post(
        "/predict",
        files={"file": (filename, img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["model_version"] == "1.0.0"
    assert data["prediction"] in ["cat", "dog"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_health_ready():
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_version"] == "1.0.0"
    assert "model_source" in data
    assert "environment" in data



def test_predict_file_too_large():
    # 10 MB + 1 byte dummy payload
    large_bytes = b"0" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/predict",
        files={
            "file": (
                "large.jpg",
                large_bytes,
                "image/jpeg",
            )
        },
    )
    assert response.status_code == 413
    assert "Image exceeds the 10 MB limit" in response.json()["detail"]


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "image_predictions_total" in response.text
    assert "image_prediction_latency_seconds_count" in response.text


def test_model_endpoint():
    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "image-classifier"
    assert data["version"] == "1.0.0"
    assert data["source"] in ["local", "mlflow"]
    assert "uri" in data
    assert "environment" in data


