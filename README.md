# End-to-End Image Classification MLOps Service

A production-grade, end-to-end MLOps solution for Image Classification. This repository features model training with MLflow tracking, high-performance model serving with FastAPI and PyTorch, continuous monitoring with Prometheus & Grafana, Kubernetes deployment manifests, Infrastructure as Code with Terraform, and automated CI/CD workflows via GitHub Actions.

---

## 📁 Repository Structure

```
├── app/
│   ├── main.py                   # FastAPI application entrypoint & middleware
│   ├── routes/
│   │   ├── predict.py            # Image inference prediction route
│   │   └── health.py             # Liveness and readiness health endpoints
│   ├── inference/
│   │   ├── predictor.py          # Model loader and inference engine
│   │   └── preprocessing.py      # Image transformations and tensor pipeline
│   └── monitoring/
│       └── metrics.py            # Prometheus custom metrics instrumentation
│
├── training/
│   ├── train.py                  # Model training pipeline with MLflow logging
│   ├── evaluate.py               # Evaluation metrics & confusion matrix
│   ├── dataset.py                # Dataset loaders and augmentations
│   └── config.yaml               # Training & experiment hyperparameters
│
├── models/                       # Model artifacts and serialized weights (.pt, .onnx)
├── tests/
│   ├── test_api.py               # FastAPI integration and endpoint tests
│   ├── test_model.py             # Model inference unit tests
│   └── test_preprocessing.py      # Image preprocessing unit tests
│
├── mlflow/                       # Local/shared MLflow tracking backend
├── k8s/
│   ├── deployment.yaml           # Kubernetes pod deployment with probes
│   ├── service.yaml              # Kubernetes ClusterIP service
│   ├── ingress.yaml              # Ingress routing rules
│   └── configmap.yaml            # Environment and runtime configuration
│
├── monitoring/
│   ├── prometheus.yml            # Prometheus scrape targets configuration
│   └── grafana/                  # Grafana dashboard & datasource provisions
│
├── terraform/
│   ├── main.tf                   # Cloud infrastructure resource definitions
│   ├── variables.tf              # Terraform input variables
│   ├── outputs.tf                # Infrastructure output values
│   └── modules/                  # Reusable Terraform modules
│
├── .github/
│   └── workflows/
│       ├── ci.yml                # CI: Linting, testing, and container build
│       └── cd.yml                # CD: Container registry push and K8s rollout
│
├── Dockerfile                    # Multi-stage container build
├── docker-compose.yml            # Local orchestration (App + Prometheus + Grafana)
├── requirements.txt              # Pinned Python dependencies
├── README.md                     # Documentation
└── .gitignore                    # Git ignore specifications
```

---

## 🚀 Quick Start

### 1. Local Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Service Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **Prometheus Metrics:** `http://localhost:8000/metrics`

---

## 🧪 Testing

Execute the automated test suite with coverage:

```bash
pytest tests/ -v --cov=app
```

---

## 🏋️ Model Training & Tracking

Configure hyperparameters in `training/config.yaml`, then run the training pipeline:

```bash
# Run training
python training/train.py --config training/config.yaml

# Evaluate model performance
python training/evaluate.py --model-path models/model.pt

# Launch MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000
```

---

## 🐳 Docker & Docker Compose

Launch the complete stack (FastAPI Service + Prometheus + Grafana):

```bash
docker-compose up --build -d
```

- **API Service:** `http://localhost:8000`
- **Prometheus:** `http://localhost:9090`
- **Grafana:** `http://localhost:3000` (Default credentials: `admin` / `admin`)

---

## ☸️ Kubernetes Deployment

Deploy the application to a Kubernetes cluster:

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

Check pod status:
```bash
kubectl get pods -l app=image-classification-service
```

---

## 🏗️ Terraform Infrastructure

Initialize and plan infrastructure provisioning:

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

---

## 🔄 CI/CD Pipeline

The `.github/workflows/` directory contains automated GitHub Actions workflows:
- **`ci.yml`**: Runs on pull requests and pushes to `main`. Checks code formatting (`flake8`, `black`), runs unit & integration tests (`pytest`), and builds the Docker container.
- **`cd.yml`**: Triggers on releases or tags to publish container images to Docker Hub / Container Registry and updates Kubernetes deployments.
