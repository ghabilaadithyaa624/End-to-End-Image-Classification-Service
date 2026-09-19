from prometheus_client import Counter, Histogram, Gauge

# Request Counters
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status"]
)

# Inference Latency
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Time spent running model prediction inference",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Overall Request Latency
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Total HTTP request processing duration",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Prediction Class Counter
PREDICTION_CLASSES = Counter(
    "prediction_classes_total",
    "Total count of predictions by predicted class label",
    ["class_name"]
)

# Active Requests Gauge
ACTIVE_REQUESTS = Gauge(
    "http_requests_active",
    "Number of active HTTP requests currently being handled"
)
