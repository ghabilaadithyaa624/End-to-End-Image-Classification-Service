import time
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status
from pydantic import BaseModel
from typing import List

from app.inference.preprocessing import preprocess_image
from app.inference.predictor import get_classifier
from app.monitoring.metrics import INFERENCE_LATENCY, PREDICTION_CLASSES

router = APIRouter(tags=["Inference"])


class PredictionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    filename: str
    top_prediction: str
    top_confidence: float
    predictions: List[PredictionItem]
    latency_seconds: float


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify an uploaded image",
    status_code=status.HTTP_200_OK
)
async def predict_image(
    file: UploadFile = File(..., description="Image file to classify (JPEG, PNG, WEBP)"),
    top_k: int = Query(default=3, ge=1, le=10, description="Number of top predictions to return")
):
    """Accepts an uploaded image file, preprocesses it, and performs deep learning classification."""
    allowed_content_types = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    if file.content_type and file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_content_types}"
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        tensor = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image: {str(e)}"
        )

    classifier = get_classifier()

    with INFERENCE_LATENCY.time():
        result = classifier.predict(tensor, top_k=top_k)

    # Track Prometheus metric for top predicted class
    top_class = result.get("top_prediction", "unknown")
    PREDICTION_CLASSES.labels(class_name=top_class).inc()

    return PredictionResponse(
        filename=file.filename or "unknown",
        top_prediction=result["top_prediction"],
        top_confidence=result["top_confidence"],
        predictions=[PredictionItem(**p) for p in result["predictions"]],
        latency_seconds=result["latency_seconds"]
    )
