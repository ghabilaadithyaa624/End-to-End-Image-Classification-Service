import urllib.request
import mlflow
import pytest


def test_mlflow_connection():

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    try:
        urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=1)
    except Exception:
        pytest.skip("MLflow tracking server not reachable at http://127.0.0.1:5000 (running in CI)")

    experiment_name = "image-classification-mlops"

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name
        )
    else:
        experiment_id = experiment.experiment_id

    assert experiment_id is not None


def test_governance_validation():
    from training.governance import validate_model

    assert validate_model(0.9721, threshold=0.95) is True
    assert validate_model(0.9400, threshold=0.95) is False


def test_model_registry_production_alias():
    try:
        urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=1)
    except Exception:
        pytest.skip("MLflow tracking server not reachable at http://127.0.0.1:5000 (running in CI)")

    client = mlflow.MlflowClient("http://127.0.0.1:5000")
    model_version = client.get_model_version_by_alias("image-classifier", "production")

    assert model_version is not None
    assert model_version.name == "image-classifier"
    assert "production" in model_version.aliases


