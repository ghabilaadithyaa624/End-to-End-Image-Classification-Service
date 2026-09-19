from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.routes.health import router as health_router
from app.routes.predict import router as predict_router
from app.routes.model import router as model_router


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

app.include_router(
    model_router
)



@app.get(
    "/metrics",
    tags=["Monitoring"],
)
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/")
def root():

    return {
        "service": "image-classification-api",
        "version": "1.0.0",
        "status": "running",
    }
