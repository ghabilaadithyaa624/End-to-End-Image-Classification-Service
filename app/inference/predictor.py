import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow.pytorch
import torch
from torchvision import models

from training.train import create_model


MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "models/image_classifier.pth",
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/image-classifier@production",
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "1.0.0",
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "local",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class Predictor:
    """
    Inference predictor supporting configurable local files
    and MLflow Model Registry sources (@production).
    """

    def __init__(self, model_path: Optional[str] = None):
        self.device = DEVICE
        self.source = os.getenv("MODEL_SOURCE", MODEL_SOURCE)
        self.model_path = model_path or os.getenv("MODEL_PATH", MODEL_PATH)
        self.model_uri = os.getenv("MODEL_URI", MODEL_URI)
        self.model_version = os.getenv("MODEL_VERSION", MODEL_VERSION)
        self.environment = os.getenv("ENVIRONMENT", ENVIRONMENT)

        if self.source == "mlflow":
            self.model = self._load_mlflow_model()
        else:
            self.model = self._load_local_model()

        if hasattr(self.model, "eval"):
            try:
                self.model.eval()
            except NotImplementedError:
                pass

        self.class_names = ["cat", "dog"]

    def _load_mlflow_model(self):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(tracking_uri)
        model = mlflow.pytorch.load_model(self.model_uri)
        return model

    def _load_local_model(self):
        target_path = Path(self.model_path)
        if not target_path.exists() and Path("models/best_model.pth").exists():
            target_path = Path("models/best_model.pth")
        if not target_path.exists() and Path("models/image_classifier.pth").exists():
            target_path = Path("models/image_classifier.pth")

        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model = create_model(num_classes=2, pretrained=False)
        loaded = torch.load(
            target_path,
            map_location=self.device,
            weights_only=True,
        )

        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        else:
            state_dict = loaded

        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def predict(self, tensor: torch.Tensor) -> Dict[str, Any]:
        tensor = tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        return {
            "prediction": self.class_names[predicted.item()],
            "confidence": round(float(confidence.item()), 4),
            "model_version": self.model_version,
        }


# Backwards compatibility alias
ImagePredictor = Predictor
