import sys
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torchvision import models

from training.dataset import create_dataloaders
from training.train import create_model


def load_config(config_path: str = "training/config.yaml"):
    """Load configuration from YAML."""

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate_model(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evaluate a trained model and return predictions,
    labels, and average loss.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = total_loss / total_samples

    return (
        average_loss,
        all_labels,
        all_predictions,
    )


def main():

    # --------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------

    config = load_config()

    # --------------------------------------------------
    # 2. Select device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------
    # 3. Load test dataset
    # --------------------------------------------------

    _, _, test_loader = create_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    class_names = test_loader.dataset.classes

    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {class_names}")

    # --------------------------------------------------
    # 4. Create model
    # --------------------------------------------------

    model = create_model(
        num_classes=config["model"]["num_classes"],
        pretrained=False,
    )

    model_path = Path(
        config["output"]["model_path"]
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run training first."
        )

    # --------------------------------------------------
    # 5. Load trained weights
    # --------------------------------------------------

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model = model.to(device)

    print(f"Loaded model: {model_path}")

    # --------------------------------------------------
    # 6. Loss function
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # 7. Evaluate
    # --------------------------------------------------

    test_loss, labels, predictions = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
    )

    # --------------------------------------------------
    # 8. Calculate metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Test Loss : {test_loss:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Accuracy  : {accuracy * 100:.2f}%")

    # --------------------------------------------------
    # 9. Classification report
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
    )

    print(report)

    # --------------------------------------------------
    # 10. Confusion matrix
    # --------------------------------------------------

    matrix = confusion_matrix(
        labels,
        predictions,
    )

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(matrix)

    print("\nRows    = Actual")
    print("Columns = Predicted")

    for i, class_name in enumerate(class_names):

        print(
            f"\n{class_name}:"
        )

        print(
            f"  Predicted {class_name}: "
            f"{matrix[i][i]}"
        )

        other_class = class_names[
            1 - i
        ]

        print(
            f"  Predicted {other_class}: "
            f"{matrix[i][1 - i]}"
        )


if __name__ == "__main__":
    main()
