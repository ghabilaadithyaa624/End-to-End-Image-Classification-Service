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
    sample_cat = next(Path("data/raw/test/cat").glob("*.jpg"))
    with open(sample_cat, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/predict",
        files={"file": (sample_cat.name, img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["model_version"] == "1.0.0"
    assert data["prediction"] in ["cat", "dog"]
    assert 0.0 <= data["confidence"] <= 1.0


