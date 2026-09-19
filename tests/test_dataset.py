import torch
from training.dataset import create_dataloaders


def test_dataset_loading():
    """Verify that dataset splits load correctly with classes and mappings."""
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="data/raw",
        batch_size=4,
        num_workers=0
    )

    assert len(train_loader.dataset) > 0, "Train dataset should not be empty"
    assert len(val_loader.dataset) > 0, "Validation dataset should not be empty"
    assert len(test_loader.dataset) > 0, "Test dataset should not be empty"

    assert train_loader.dataset.classes == ["cat", "dog"]
    assert train_loader.dataset.class_to_idx == {"cat": 0, "dog": 1}


def test_image_shape():
    """Verify that a batch of images has shape [B, 3, 224, 224] and labels [B]."""
    batch_size = 4
    train_loader, _, _ = create_dataloaders(
        data_dir="data/raw",
        batch_size=batch_size,
        num_workers=0
    )

    images, labels = next(iter(train_loader))

    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.shape == torch.Size([batch_size, 3, 224, 224])
    assert labels.shape == torch.Size([batch_size])
