from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.inference.predictor import get_classifier

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live", summary="Liveness Probe", status_code=status.HTTP_200_OK)
async def liveness():
    """Liveness probe to confirm that the server process is alive."""
    return {"status": "alive"}


@router.get("/ready", summary="Readiness Probe", status_code=status.HTTP_200_OK)
async def readiness():
    """Readiness probe to confirm that model weights and resources are loaded."""
    try:
        classifier = get_classifier()
        if classifier.model is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "Model is not loaded"}
            )
        return {
            "status": "ready",
            "model_path": classifier.model_path,
            "device": str(classifier.device)
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": str(e)}
        )
