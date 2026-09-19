from prometheus_client import Counter, Histogram


PREDICTIONS_TOTAL = Counter(
    "image_predictions_total",
    "Total number of image predictions",
    ["prediction"],
)


PREDICTION_ERRORS_TOTAL = Counter(
    "image_prediction_errors_total",
    "Total number of prediction errors",
)


PREDICTION_LATENCY = Histogram(
    "image_prediction_latency_seconds",
    "Time spent performing image predictions",
)


HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)
