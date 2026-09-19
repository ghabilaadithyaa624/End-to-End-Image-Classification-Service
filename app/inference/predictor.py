import os
import time
import logging
from typing import Dict, List, Any, Optional
import torch
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)

# Default ImageNet 10-class sample fallback labels
DEFAULT_CLASSES = [
    "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
    "electric_ray", "stingray", "cock", "hen", "ostrich"
]


class ImageClassifier:
    """Manages PyTorch model loading and inference execution."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        default_path = "models/image_classifier.pth" if os.path.exists("models/image_classifier.pth") else "models/model.pt"
        self.model_path = model_path or os.getenv("MODEL_PATH", default_path)
        self.classes: List[str] = DEFAULT_CLASSES
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """Loads a model from path or initializes a standard pre-trained MobileNetV3."""
        if os.path.exists(self.model_path):
            try:
                logger.info(f"Loading custom model from {self.model_path} onto {self.device}")
                loaded = torch.load(self.model_path, map_location=self.device, weights_only=True)
                if isinstance(loaded, dict):
                    # ResNet18 binary classifier
                    model = models.resnet18(weights=None)
                    model.fc = torch.nn.Linear(model.fc.in_features, 2)
                    state_dict = loaded.get("model_state_dict", loaded)
                    model.load_state_dict(state_dict)
                    self.classes = ["cat", "dog"]
                else:
                    model = loaded
                model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                logger.warning(f"Failed to load model from {self.model_path}: {e}. Falling back to default architecture.")

        logger.info(f"Initializing MobileNetV3-Small architecture on {self.device}")
        try:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            # Fallback for offline environments without pre-downloaded weights
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(self.classes))

        model.to(self.device)
        model.eval()
        return model

    def predict(self, input_tensor: torch.Tensor, top_k: int = 3) -> Dict[str, Any]:
        """Runs inference on a preprocessed input tensor and returns predictions with confidence."""
        start_time = time.perf_counter()
        tensor = input_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)

        num_classes = outputs.shape[1]
        k = min(top_k, num_classes)
        top_probs, top_indices = torch.topk(probabilities, k=k, dim=1)

        top_probs_list = top_probs[0].cpu().tolist()
        top_indices_list = top_indices[0].cpu().tolist()

        predictions = []
        for prob, idx in zip(top_probs_list, top_indices_list):
            label = self.classes[idx] if idx < len(self.classes) else f"class_{idx}"
            predictions.append({
                "class_id": int(idx),
                "class_name": label,
                "confidence": round(float(prob), 4)
            })

        latency = time.perf_counter() - start_time

        return {
            "top_prediction": predictions[0]["class_name"] if predictions else "unknown",
            "top_confidence": predictions[0]["confidence"] if predictions else 0.0,
            "predictions": predictions,
            "latency_seconds": round(latency, 4)
        }


# Global singleton instance
classifier_instance: Optional[ImageClassifier] = None


def get_classifier() -> ImageClassifier:
    """Returns the singleton classifier instance."""
    global classifier_instance
    if classifier_instance is None:
        classifier_instance = ImageClassifier()
    return classifier_instance
