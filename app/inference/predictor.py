from pathlib import Path
from typing import Dict, Any, Optional
import torch

from training.train import create_model


class ImagePredictor:
    """
    Loads the trained image classification model
    and performs inference.
    """

    def __init__(
        self,
        model_path: str = "models/image_classifier.pth",
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = create_model(
            num_classes=2,
            pretrained=False,
        )

        loaded = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        else:
            state_dict = loaded

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(self.device)

        self.model.eval()

        self.class_names = [
            "cat",
            "dog",
        ]

        self.model_version = "1.0.0"

    def predict(
        self,
        image_tensor: torch.Tensor,
    ) -> Dict[str, Any]:

        image_tensor = image_tensor.to(
            self.device
        )

        with torch.no_grad():

            outputs = self.model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            confidence, predicted_index = (
                torch.max(
                    probabilities,
                    dim=1,
                )
            )

        predicted_class = self.class_names[
            predicted_index.item()
        ]

        confidence_value = round(confidence.item(), 4)

        return {
            "prediction": predicted_class,
            "confidence": confidence_value,
            "model_version": self.model_version,
        }

