import os
import mlflow
import mlflow.pytorch
import torch

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_uri)

MODEL_URI = os.getenv("MODEL_URI", "models:/image-classifier@production")

model = mlflow.pytorch.load_model(MODEL_URI)

if hasattr(model, "eval"):
    try:
        model.eval()
    except NotImplementedError:
        pass

print("Production model loaded successfully")
print(model)
