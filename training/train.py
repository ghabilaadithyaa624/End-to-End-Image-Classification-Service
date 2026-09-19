import os
import argparse
import yaml
import torch
import torch.nn as nn
from torchvision import models
import mlflow
import mlflow.pytorch
from training.dataset import get_dataloaders


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Constructs the vision model with a customized classifier head."""
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Image Classification Model")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["project"]["experiment_name"])

    # Dataloaders
    train_loader, val_loader = get_dataloaders(
        train_dir=cfg["data"]["train_data_dir"],
        val_dir=cfg["data"]["val_data_dir"],
        batch_size=cfg["data"]["batch_size"],
        image_size=tuple(cfg["data"]["image_size"]),
        num_workers=0  # Safe cross-platform default
    )

    model = build_model(
        architecture=cfg["model"]["architecture"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"]
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["hyperparameters"]["learning_rate"],
        weight_decay=cfg["hyperparameters"]["weight_decay"]
    )

    with mlflow.start_run():
        mlflow.log_params({
            "architecture": cfg["model"]["architecture"],
            "epochs": cfg["hyperparameters"]["epochs"],
            "learning_rate": cfg["hyperparameters"]["learning_rate"],
            "batch_size": cfg["data"]["batch_size"],
            "optimizer": cfg["hyperparameters"]["optimizer"],
        })

        best_acc = 0.0
        save_path = cfg["model"]["save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(1, cfg["hyperparameters"]["epochs"] + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            print(f"Epoch [{epoch}/{cfg['hyperparameters']['epochs']}] - "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model, save_path)
                print(f"--> Saved best model checkpoint to {save_path}")

        # Save final model
        if not os.path.exists(save_path):
            torch.save(model, save_path)

        if cfg["mlflow"].get("log_models", True):
            mlflow.pytorch.log_model(model, artifact_path="model")
            print("Model logged to MLflow artifacts.")


if __name__ == "__main__":
    main()
