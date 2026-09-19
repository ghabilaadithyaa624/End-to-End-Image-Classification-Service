from fastapi import FastAPI

from app.routes.health import router as health_router
from app.routes.predict import router as predict_router


app = FastAPI(
    title="Image Classification API",
    description=(
        "Production-oriented image classification "
        "service using ResNet18."
    ),
    version="1.0.0",
)


app.include_router(
    health_router
)

app.include_router(
    predict_router
)


@app.get("/")
def root():

    return {
        "service": "image-classification-api",
        "version": "1.0.0",
        "status": "running",
    }

