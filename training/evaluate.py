import argparse
import os
import torch
from sklearn.metrics import classification_report, confusion_matrix
from training.dataset import get_dataloaders
from training.train import load_config


def evaluate(model_path: str, config_path: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"Loading model checkpoint from {model_path} onto {device}...")
    loaded = torch.load(model_path, map_location=device)
    if isinstance(loaded, dict):
        from training.train import create_model
        model = create_model(num_classes=cfg["model"]["num_classes"], pretrained=False)
        state_dict = loaded.get("model_state_dict", loaded)
        model.load_state_dict(state_dict)
    else:
        model = loaded
    model.to(device)
    model.eval()

    eval_dir = cfg["data"].get("test_data_dir") or cfg["data"]["val_data_dir"]
    print(f"Evaluating model on dataset: {eval_dir}")

    _, eval_loader = get_dataloaders(
        train_dir=cfg["data"]["train_data_dir"],
        val_dir=eval_dir,
        batch_size=cfg["data"]["batch_size"],
        image_size=tuple(cfg["data"]["image_size"]),
        num_workers=0
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_targets, all_preds, zero_division=0))

    print("CONFUSION MATRIX")
    print("="*50)
    print(confusion_matrix(all_targets, all_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Classification Model")
    parser.add_argument("--model-path", type=str, default="models/model.pt", help="Path to saved model")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config file")
    args = parser.parse_args()

    evaluate(args.model_path, args.config)
