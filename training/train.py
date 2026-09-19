import sys
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows console supports UTF-8 (emojis from MLflow logger like \U0001f3c3)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torchvision import models

from training.dataset import create_dataloaders
from training.utils import get_git_commit


def load_config(config_path: str = "training/config.yaml"):
    """Load training configuration from YAML."""

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_model(num_classes: int, pretrained: bool = True):
    """Create a ResNet18 model for binary classification."""

    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # Freeze the pretrained backbone initially.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the final classification layer.
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """Train the model for one epoch."""

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(
    model,
    dataloader,
    criterion,
    device,
):
    """Evaluate the model on validation data."""

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def main():

    # --------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------

    config = load_config()

    # --------------------------------------------------
    # 2. Configure MLflow
    # --------------------------------------------------

    mlflow.set_tracking_uri(
        config.get("mlflow", {}).get("tracking_uri", "http://127.0.0.1:5000")
    )

    mlflow.set_experiment(
        config["project"]["name"]
    )

    # --------------------------------------------------
    # 3. Select device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("IMAGE CLASSIFICATION TRAINING (MLflow Tracking Enabled)")
    print("=" * 60)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"Device          : {device}")

    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(0)}")

    print(f"MLflow URI      : {mlflow.get_tracking_uri()}")
    print(f"MLflow Exp      : {config['project']['name']}")
    print("=" * 60)

    # --------------------------------------------------
    # 4. Create DataLoaders
    # --------------------------------------------------

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    print(f"Training images   : {len(train_loader.dataset)}")
    print(f"Validation images : {len(val_loader.dataset)}")
    print(f"Test images       : {len(test_loader.dataset)}")

    print(
        f"Classes           : {train_loader.dataset.classes}"
    )

    # --------------------------------------------------
    # 5. Create model & optimizer
    # --------------------------------------------------

    model = create_model(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs = config["training"]["epochs"]
    best_val_accuracy = 0.0

    model_path = Path(config["output"]["model_path"])
    checkpoint_path = Path(config["output"]["checkpoint_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 6. Wrapped in MLflow Run
    # --------------------------------------------------

    with mlflow.start_run():

        # Log training parameters
        mlflow.log_params(
            {
                "model": config["model"]["name"],
                "epochs": config["training"]["epochs"],
                "batch_size": config["data"]["batch_size"],
                "learning_rate": config["training"]["learning_rate"],
                "weight_decay": config["training"]["weight_decay"],
                "image_size": config["data"]["image_size"],
                "optimizer": "Adam",
                "device": str(device),
                "git_commit": get_git_commit(),
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "test_samples": len(test_loader.dataset),
            }
        )

        print("\nStarting training with MLflow tracking...\n")

        for epoch in range(epochs):

            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            val_loss, val_accuracy = validate(
                model,
                val_loader,
                criterion,
                device,
            )

            # Log metrics for every epoch to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch + 1,
            )

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_accuracy:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_accuracy:.4f}"
            )

            # Save best model checkpoint locally
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

                torch.save(
                    model.state_dict(),
                    model_path,
                )

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_accuracy,
                    },
                    checkpoint_path,
                )

                print(
                    f"  --> Best model saved "
                    f"(val accuracy: {val_accuracy:.4f})"
                )

        # Governance tags
        mlflow.set_tag("model_type", "resnet18")
        mlflow.set_tag("task", "binary_image_classification")
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("validation_status", "candidate")

        # Log the best model artifact
        mlflow.log_metric(
            "best_val_accuracy",
            best_val_accuracy,
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model_files",
        )

        # Register model in MLflow Model Registry
        print("Registering model in MLflow Model Registry...")
        input_example = torch.randn(1, 3, 224, 224, device=device)
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name="image-classifier",
            input_example=input_example.cpu().numpy(),
        )



    print("\n" + "=" * 60)
    print("TRAINING COMPLETE (Tracked in MLflow)")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
