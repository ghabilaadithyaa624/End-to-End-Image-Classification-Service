import os
from fastapi import APIRouter

from app.routes.predict import predictor


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "1.0.0",
)

MODEL_SOURCE = os.getenv(
    "MODEL_SOURCE",
    "local",
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development",
)


@router.get("")
def health():
    return {
        "status": "healthy"
    }


@router.get("/live")
def liveness():
    return {
        "status": "alive"
    }


@router.get("/ready")
def readiness():
    if predictor.model is None:
        return {
            "status": "not_ready"
        }

    return {
        "status": "ready",
        "model_version": os.getenv("MODEL_VERSION", getattr(predictor, "model_version", MODEL_VERSION)),
        "model_source": os.getenv("MODEL_SOURCE", getattr(predictor, "source", MODEL_SOURCE)),
        "environment": os.getenv("ENVIRONMENT", getattr(predictor, "environment", ENVIRONMENT)),
    }
