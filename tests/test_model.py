import torch
from training.train import create_model
from app.inference.predictor import ImagePredictor


def test_model_output_shape():
    model = create_model(
        num_classes=2,
        pretrained=False,
    )

    model.eval()

    sample = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        output = model(sample)

    assert output.shape == (2, 2)


def test_predictor_initialization():
    predictor = ImagePredictor()
    assert predictor.model is not None
    assert predictor.class_names == ["cat", "dog"]


def test_predictor_predict_shape_and_keys():
    predictor = ImagePredictor()
    dummy_input = torch.randn(1, 3, 224, 224)
    result = predictor.predict(dummy_input)

    assert "prediction" in result
    assert "confidence" in result
    assert "model_version" in result
    assert result["prediction"] in ["cat", "dog"]
    assert 0.0 <= result["confidence"] <= 1.0

