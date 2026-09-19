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


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


predictor = ImagePredictor()


@router.post("")
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
