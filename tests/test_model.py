import torch
from app.inference.predictor import ImageClassifier


def test_classifier_initialization():
    classifier = ImageClassifier(device="cpu")
    assert classifier.model is not None
    assert str(classifier.device) == "cpu"


def test_classifier_predict_shape_and_keys():
    classifier = ImageClassifier(device="cpu")
    dummy_input = torch.randn(1, 3, 224, 224)
    result = classifier.predict(dummy_input, top_k=3)

    assert "top_prediction" in result
    assert "top_confidence" in result
    assert "predictions" in result
    assert "latency_seconds" in result
    assert len(result["predictions"]) <= 3
    assert result["top_confidence"] >= 0.0


def test_classifier_probabilities_sum():
    classifier = ImageClassifier(device="cpu")
    dummy_input = torch.randn(1, 3, 224, 224)
    result = classifier.predict(dummy_input, top_k=5)

    for item in result["predictions"]:
        assert 0.0 <= item["confidence"] <= 1.0
        assert isinstance(item["class_name"], str)
