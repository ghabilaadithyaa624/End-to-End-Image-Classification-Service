import io
import pytest
from PIL import Image
import torch
from app.inference.preprocessing import (
    load_image_from_bytes,
    preprocess_image,
    get_inference_transforms
)


def create_sample_image_bytes(format="JPEG", size=(100, 100), color=(255, 0, 0)) -> bytes:
    """Helper to generate in-memory dummy image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


def test_load_image_from_valid_bytes():
    image_bytes = create_sample_image_bytes(format="PNG")
    image = load_image_from_bytes(image_bytes)
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"


def test_load_image_from_invalid_bytes():
    with pytest.raises(ValueError, match="Invalid or corrupted"):
        load_image_from_bytes(b"not_an_image_data")


def test_load_image_empty_bytes():
    with pytest.raises(ValueError, match="Empty image"):
        load_image_from_bytes(b"")


def test_preprocess_image_output_shape():
    image_bytes = create_sample_image_bytes(size=(300, 200))
    tensor = preprocess_image(image_bytes, image_size=(224, 224))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)


def test_inference_transforms():
    transforms_fn = get_inference_transforms((128, 128))
    img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    output_tensor = transforms_fn(img)
    assert output_tensor.shape == (3, 128, 128)
