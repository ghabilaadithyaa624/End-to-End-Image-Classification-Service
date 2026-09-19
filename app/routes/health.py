from fastapi import APIRouter

from app.routes.predict import predictor


router = APIRouter(
    prefix="/health",
    tags=["Health"],
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
        "model_version": predictor.model_version,
    }


