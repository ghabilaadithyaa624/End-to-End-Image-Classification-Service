from io import BytesIO
from typing import Tuple, Optional
from PIL import Image
import torch
from torchvision import transforms


IMAGE_SIZE = 224

CLASS_NAMES = [
    "cat",
    "dog",
]


def get_inference_transform(image_size: Optional[Tuple[int, int]] = None):
    """
    Return the same preprocessing used during
    validation and testing.
    """
    size = image_size if image_size is not None else (IMAGE_SIZE, IMAGE_SIZE)
    return transforms.Compose(
        [
            transforms.Resize(
                size
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# Alias for compatibility with test suites
get_inference_transforms = get_inference_transform


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Reads raw image bytes and converts to an RGB PIL Image."""
    if not image_bytes:
        raise ValueError("Empty image byte content provided.")
    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image format: {str(e)}")


def preprocess_image(
    image_bytes: bytes,
    image_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Convert uploaded image bytes into a model-ready tensor.
    """
    if not image_bytes:
        raise ValueError("Empty image byte content provided.")

    try:
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image format: {str(e)}")

    transform = get_inference_transform(image_size=image_size)

    tensor = transform(image)

    # Add batch dimension:
    # [3, 224, 224]
    #      ↓
    # [1, 3, 224, 224]
    tensor = tensor.unsqueeze(0)

    return tensor
