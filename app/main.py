import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.routes.health import router as health_router
from app.routes.predict import router as predict_router
from app.inference.predictor import get_classifier
from app.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY, ACTIVE_REQUESTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan events."""
    logger.info("Initializing application and warming up model...")
    # Preload the model in memory on startup
    get_classifier()
    logger.info("Model loaded successfully. Ready to serve traffic.")
    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title="Image Classification MLOps Service",
    description="Production-grade Image Classification API with monitoring, containerization, and K8s support.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Middleware to measure request latency, status codes, and active traffic."""
    ACTIVE_REQUESTS.inc()
    start_time = time.perf_counter()
    endpoint = request.url.path

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        status_code = 500
        raise exc from None
    finally:
        duration = time.perf_counter() - start_time
        ACTIVE_REQUESTS.dec()
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=status_code
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(duration)


# Mount routers
app.include_router(health_router)
app.include_router(predict_router)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint providing service metadata."""
    return {
        "service": "Image Classification MLOps Service",
        "version": "1.0.0",
        "docs_url": "/docs",
        "metrics_url": "/metrics",
        "health_url": "/health/ready"
    }


@app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
async def metrics():
    """Exposes Prometheus metrics for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
