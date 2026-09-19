import mlflow

MODEL_NAME = "image-classifier"
MODEL_VERSION = 1

# Connect to local MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.MlflowClient("http://127.0.0.1:5000")

client.set_registered_model_alias(
    MODEL_NAME,
    "production",
    MODEL_VERSION,
)

print(
    f"{MODEL_NAME} version {MODEL_VERSION} "
    "is now assigned to @production"
)
