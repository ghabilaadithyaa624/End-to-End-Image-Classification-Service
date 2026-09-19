import os

from fastapi import APIRouter


router = APIRouter(
    prefix="/model",
    tags=["Model"],
)


@router.get("")
def model_info():
    return {
        "name": "image-classifier",
        "version": os.getenv(
            "MODEL_VERSION",
            "1.0.0",
        ),
        "source": os.getenv(
            "MODEL_SOURCE",
            "local",
        ),
        "uri": os.getenv(
            "MODEL_URI",
            "models:/image-classifier@production",
        ),
        "environment": os.getenv(
            "ENVIRONMENT",
            "development",
        ),
    }
