#!/bin/sh
set -e

exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 1 \
  --allowed-hosts "*" \
  --cors-allowed-origins "*" \
  --backend-store-uri "postgresql://mlflow:${POSTGRES_PASSWORD}@postgres.mlops.svc.cluster.local:5432/mlflow" \
  --default-artifact-root "s3://image-classification-mlops-dev-mlflow-djntbnf0/mlflow"
