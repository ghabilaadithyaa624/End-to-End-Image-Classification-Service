import io
from typing import Tuple
from PIL import Image
import torch
from torchvision import transforms

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Returns the transformation pipeline used for model inference."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Reads raw image bytes and converts to an RGB PIL Image."""
    if not image_bytes:
        raise ValueError("Empty image byte content provided.")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Verify image integrity
        # Re-open because verify leaves the image closed
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image format: {str(e)}")


def preprocess_image(image_bytes: bytes, image_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Loads, validates, and preprocesses raw image bytes into a model-ready batch tensor."""
    image = load_image_from_bytes(image_bytes)
    pipeline = get_inference_transforms(image_size=image_size)
    tensor = pipeline(image)
    return tensor.unsqueeze(0)  # Shape: (1, C, H, W)
