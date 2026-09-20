"""
End-to-End Image Classification Service - Gradio Web Application
================================================================
Production-grade interactive web interface for Cat vs Dog image classification
powered by PyTorch ResNet18, MLflow tracking, and Prometheus monitoring.
"""

import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows console supports UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Silence MLflow agent hints in console
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from PIL import Image
import torch
import gradio as gr

from app.inference.predictor import Predictor
from app.inference.preprocessing import preprocess_image
from app.monitoring.metrics import (
    PREDICTIONS_TOTAL,
    PREDICTION_ERRORS_TOTAL,
    PREDICTION_LATENCY,
)

# Initialize Predictor singleton
try:
    predictor = Predictor()
    model_loaded = True
    model_load_error = None
except Exception as exc:
    predictor = None
    model_loaded = False
    model_load_error = str(exc)


def classify_image(
    image: Optional[Image.Image],
    threshold: float = 70.0,
) -> Tuple[Dict[str, float], str, str]:
    """
    Classify an uploaded image as Cat or Dog.
    Applies an uncertainty threshold: if top class confidence is below
    the threshold, flags the image as potentially not a cat or dog.

    Returns:
        - Dict of class names to probabilities (for gr.Label)
        - Formatted Markdown summary
        - Status badge string
    """
    if image is None:
        return {}, "⚠️ **No image provided.** Please upload or snap an image.", "Waiting for input..."

    if not model_loaded or predictor is None:
        return (
            {},
            f"❌ **Model Error:** Failed to load model: {model_load_error}",
            "Model unavailable",
        )

    start_time = time.perf_counter()

    try:
        # Convert PIL Image to RGB bytes
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        # Preprocess to tensor
        tensor = preprocess_image(image_bytes)
        tensor = tensor.to(predictor.device)

        # Inference
        with torch.no_grad():
            outputs = predictor.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Record Prometheus metrics if available
        try:
            pred_name = predictor.class_names[predicted_idx.item()]
            PREDICTIONS_TOTAL.labels(prediction=pred_name).inc()
            PREDICTION_LATENCY.observe(latency_ms / 1000.0)
        except Exception:
            pass

        # Build class probabilities dictionary for gr.Label
        class_probs = {
            predictor.class_names[i].capitalize(): round(float(probabilities[i].item()), 4)
            for i in range(len(predictor.class_names))
        }

        top_class = predictor.class_names[predicted_idx.item()].capitalize()
        conf_percent = float(confidence.item()) * 100
        icon = "🐱" if top_class.lower() == "cat" else "🐶"

        is_uncertain = conf_percent < float(threshold)

        if is_uncertain:
            summary_md = f"""
### ⚠️ Prediction: **Uncertain** (Top guess: {top_class} at `{conf_percent:.1f}%`)

> 🔍 **Out-of-Domain / Non-Cat-Dog Warning:**
> The model confidence (`{conf_percent:.1f}%`) is lower than the **{threshold:.0f}%** certainty threshold. 
> This image may **not be a cat or dog**, or the subject may be obstructed or ambiguous.

- **Top Candidate:** {icon} `{top_class}` (`{conf_percent:.2f}%`)
- **Certainty Threshold:** `{threshold:.0f}%`
- **Inference Latency:** `{latency_ms:.1f} ms`
- **Model Version:** `{predictor.model_version}`
- **Device:** `{predictor.device}`
"""
            status_text = f"⚠️ Uncertain ({conf_percent:.1f}% < {threshold:.0f}% threshold): May not be a cat or dog"
        else:
            summary_md = f"""
### {icon} Prediction: **{top_class}** (Confident)
- **Confidence Score:** `{conf_percent:.2f}%` (Meets `{threshold:.0f}%` threshold)
- **Inference Latency:** `{latency_ms:.1f} ms`
- **Model Version:** `{predictor.model_version}`
- **Device:** `{predictor.device}`
- **Model Source:** `{predictor.source}`
"""
            status_text = f"✅ Confident: Classified as {top_class} ({conf_percent:.1f}%) in {latency_ms:.1f}ms"

        return class_probs, summary_md, status_text

    except Exception as exc:
        try:
            PREDICTION_ERRORS_TOTAL.inc()
        except Exception:
            pass
        error_msg = f"❌ **Inference Error:** {str(exc)}"
        return {}, error_msg, f"Error: {str(exc)}"


def get_system_health() -> str:
    """Return runtime system health and configuration status."""
    device_name = str(predictor.device) if predictor else "N/A"
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU only (No CUDA GPU detected)"

    return f"""
| Metric / Component | Status / Detail |
| :--- | :--- |
| **Model Loaded** | {'✅ Ready' if model_loaded else '❌ Error'} |
| **Architecture** | ResNet-18 (Binary Classification) |
| **Classes** | `cat`, `dog` |
| **Active Device** | `{device_name}` (`{gpu_name}`) |
| **Model Source** | `{predictor.source if predictor else 'N/A'}` |
| **Model Path** | `{predictor.model_path if predictor else 'N/A'}` |
| **Model Version** | `{predictor.model_version if predictor else 'N/A'}` |
| **PyTorch Version** | `{torch.__version__}` |
| **Host Environment** | `{sys.platform}` |
"""


# Prepare example images
sample_examples: List[List[Any]] = []
cat_sample = PROJECT_ROOT / "data" / "raw" / "test" / "cat" / "cat_00000.jpg"
dog_sample = PROJECT_ROOT / "data" / "raw" / "test" / "dog" / "dog_00000.jpg"
if cat_sample.exists():
    sample_examples.append([str(cat_sample), 70])
if dog_sample.exists():
    sample_examples.append([str(dog_sample), 70])


# Build Gradio UI
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

with gr.Blocks() as demo:
    gr.Markdown(
        """
# 🐾 End-to-End Image Classification Service
### Production MLOps Serving Interface • ResNet18 • PyTorch • Pinokio Ready
Classify images of cats and dogs with real-time confidence scores, latency telemetry, and uncertainty detection.
        """
    )

    with gr.Tabs():
        with gr.TabItem("🚀 Live Inference"):
            with gr.Row():
                with gr.Column(scale=5):
                    input_image = gr.Image(
                        type="pil",
                        label="Upload Image (Cat or Dog)",
                        sources=["upload", "clipboard", "webcam"],
                    )
                    threshold_slider = gr.Slider(
                        minimum=50,
                        maximum=95,
                        value=70,
                        step=5,
                        label="🎯 Certainty Threshold (%)",
                        info="Predictions below this score will trigger an 'Uncertain / Not Cat or Dog' alert.",
                    )
                    with gr.Row():
                        submit_btn = gr.Button("⚡ Run Classification", variant="primary", scale=2)
                        clear_btn = gr.ClearButton(scale=1)

                with gr.Column(scale=5):
                    label_output = gr.Label(
                        num_top_classes=2,
                        label="Classification Probabilities",
                    )
                    summary_output = gr.Markdown("Prediction details will appear here...")
                    status_output = gr.Textbox(
                        label="Serving Status",
                        value="System ready. Upload an image to test.",
                        interactive=False,
                    )

            if sample_examples:
                gr.Examples(
                    examples=sample_examples,
                    inputs=[input_image, threshold_slider],
                    outputs=[label_output, summary_output, status_output],
                    fn=classify_image,
                    cache_examples=False,
                    label="Sample Benchmark Images",
                )

            submit_btn.click(
                fn=classify_image,
                inputs=[input_image, threshold_slider],
                outputs=[label_output, summary_output, status_output],
            )
            input_image.change(
                fn=classify_image,
                inputs=[input_image, threshold_slider],
                outputs=[label_output, summary_output, status_output],
            )
            threshold_slider.change(
                fn=classify_image,
                inputs=[input_image, threshold_slider],
                outputs=[label_output, summary_output, status_output],
            )
            clear_btn.add([input_image, label_output, summary_output, status_output])

        with gr.TabItem("🩺 System Health & Specs"):
            gr.Markdown("### Runtime Diagnostics & Model Provenance")
            health_table = gr.Markdown(value=get_system_health())
            refresh_health_btn = gr.Button("🔄 Refresh System Health")
            refresh_health_btn.click(fn=get_system_health, outputs=[health_table])

        with gr.TabItem("📖 MLOps Architecture"):
            gr.Markdown(
                """
### 🏗️ Architecture & Pipeline Details
- **Training Pipeline:** ResNet-18 fine-tuned on Cats vs Dogs dataset with MLflow experiment tracking.
- **Inference Engine:** PyTorch high-performance tensor evaluation with standard normalization `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
- **FastAPI Model Server:** Also available via `uvicorn app.main:app --port 8000` for REST API consumption.
- **Monitoring & Metrics:** Integrated Prometheus metrics (`PREDICTIONS_TOTAL`, `PREDICTION_LATENCY`, `PREDICTION_ERRORS_TOTAL`) scrapable at `/metrics`.
- **Pinokio Integration:** Autostart configured via `pinokio.js`, `install.json`, and `start.json` on port 7860.
                """
            )

    gr.Markdown(
        """
---
<center>
<small>End-to-End Image Classification Service | Designed for Local, Cloud & Pinokio Deployment</small>
</center>
        """
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 7860))

    # Print local URL explicitly for Pinokio process listener
    print(f"Starting Gradio server at http://{host}:{port}")

    demo.launch(
        server_name=host,
        server_port=port,
        theme=custom_theme,
        share=False,
    )
