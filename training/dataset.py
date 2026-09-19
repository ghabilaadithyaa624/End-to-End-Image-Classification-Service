import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Returns data augmentation and normalization transforms for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Returns validation/test transforms without augmentations."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class SyntheticImageDataset(Dataset):
    """Synthetic dataset for testing and dry-run training pipelines."""

    def __init__(self, num_samples: int = 100, num_classes: int = 10, image_size: Tuple[int, int] = (224, 224)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = torch.randn(3, self.image_size[0], self.image_size[1])
        label = idx % self.num_classes
        return image, label


def get_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """Builds training and validation DataLoaders with fallback to synthetic data for testing."""
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms(image_size))
        val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transforms(image_size))
    else:
        # Fallback to synthetic data for validation and testing
        train_dataset = SyntheticImageDataset(num_samples=128, image_size=image_size)
        val_dataset = SyntheticImageDataset(num_samples=32, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
