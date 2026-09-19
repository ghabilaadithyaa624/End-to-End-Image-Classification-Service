from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.inference.preprocessing import (
    preprocess_image,
)
from app.inference.predictor import (
    ImagePredictor,
)
from app.schemas.prediction import PredictionResponse


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


predictor = ImagePredictor()


@router.post(
    "",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...)
):
    """
    Classify an uploaded image.
    """

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type.",
        )

    if not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="File must be an image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit.",
        )

    try:

        image_tensor = preprocess_image(
            image_bytes
        )

        result = predictor.predict(
            image_tensor
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {exc}",
        ) from exc

