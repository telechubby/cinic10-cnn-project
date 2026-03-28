"""
Data Preprocessing Module for CINIC-10 Dataset

This module handles the preprocessing of the CINIC-10 dataset for CNN image classification.
It includes data loading, augmentation, and preparation for model training.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

# CINIC-10 class labels
CINIC_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_cinic_data(data_dir, subset="train"):
    """
    Load CINIC-10 data from directory structure.

    Args:
        data_dir (str): Path to the CINIC-10 dataset directory
        subset (str): Which subset to load ('train', 'valid', 'test')

    Returns:
        tuple: (images, labels) numpy arrays
    """
    # Define paths for different subsets
    if subset == "train":
        data_path = os.path.join(data_dir, "train")
    elif subset == "valid":
        data_path = os.path.join(data_dir, "valid")
    elif subset == "test":
        data_path = os.path.join(data_dir, "test")
    else:
        raise ValueError("Subset must be 'train', 'valid', or 'test'")

    # Collect all image paths and labels
    image_paths = []
    labels = []

    # Iterate through class directories
    for class_idx, class_name in enumerate(CINIC_CLASSES):
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            for filename in os.listdir(class_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(class_path, filename))
                    labels.append(class_idx)

    # Load images
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            # Resize to 32x32 as required by CINIC-10
            img = img.resize((32, 32))
            images.append(np.array(img))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels)


def create_data_generators(train_dir, validation_dir, batch_size=32, augment=True):
    """
    Create DataLoaders for training and validation.

    Returns:
        tuple: (train_loader, val_loader) -- torch DataLoader instances
               Batches yield (images: FloatTensor (B,3,32,32), labels: LongTensor (B,))
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1), shear=5.7),  # 0.1 rad ≈ 5.7°
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(validation_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4,
                              persistent_workers=True, pin_memory=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4,
                              persistent_workers=True, pin_memory=False)
    return train_loader, val_loader


# CINIC-10 channel statistics (computed over the full 270k-image dataset)
CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC10_STD  = (0.24205776, 0.23828046, 0.25874022)


def create_data_generators_strong(train_dir, validation_dir, batch_size=512, num_workers=4):
    """
    DataLoaders with strong augmentation for VGG/ResNet-style training on 32×32 images.

    Train pipeline:
        RandomCrop(32, padding=4)          — standard ±4px translation for CIFAR-scale images
        RandomHorizontalFlip()
        AutoAugment(CIFAR10)               — policy learned on CIFAR-10; safe magnitudes for 32×32
        ToTensor()
        Normalize(CINIC-10 mean, std)

    Val/test pipeline: Resize → ToTensor → Normalize only (no stochastic transforms).

    Why AutoAugment(CIFAR10) over TrivialAugmentWide
    -------------------------------------------------
    TrivialAugmentWide uses "widest possible" magnitude ranges (Rotate ±135°,
    TranslateX/Y up to 32 px, ShearX/Y up to 0.99).  On 32×32 pixels those
    extremes completely destroy image content, collapsing first-epoch accuracy
    from ~40% to ~23%.  AutoAugment(CIFAR10) uses a policy optimised for small
    images and applies moderate, content-preserving magnitudes.

    Notes
    -----
    * AutoAugmentPolicy.CIFAR10 is available in torchvision >= 0.12 (torch >= 1.11).
    * Normalize shifts inputs to roughly zero-mean unit-variance, which is
      important for SGD + weight-decay and stabilises BN statistics.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CINIC10_MEAN, CINIC10_STD),
    ])

    train_dataset = datasets.ImageFolder(train_dir,      transform=train_transform)
    val_dataset   = datasets.ImageFolder(validation_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              persistent_workers=True, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              persistent_workers=True, pin_memory=False)
    return train_loader, val_loader


def get_cinic_statistics(data_dir):
    """
    Get statistics about the CINIC-10 dataset.

    Args:
        data_dir (str): Path to the CINIC-10 dataset directory

    Returns:
        dict: Dataset statistics
    """
    stats = {"classes": CINIC_CLASSES, "class_counts": {}, "total_images": 0}

    for class_idx, class_name in enumerate(CINIC_CLASSES):
        class_path = os.path.join(data_dir, "train", class_name)
        if os.path.exists(class_path):
            count = len(os.listdir(class_path))
            stats["class_counts"][class_name] = count
            stats["total_images"] += count

    return stats


def visualize_class_distribution(data_dir, save_path=None):
    """
    Visualize the distribution of classes in CINIC-10 dataset.

    Args:
        data_dir (str): Path to the CINIC-10 dataset directory
        save_path (str): Path to save the visualization (optional)
    """
    stats = get_cinic_statistics(data_dir)

    plt.figure(figsize=(12, 6))
    classes = list(stats["class_counts"].keys())
    counts = list(stats["class_counts"].values())

    bars = plt.bar(range(len(classes)), counts, color="skyblue")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.title("Distribution of CINIC-10 Dataset Classes")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def prepare_cinic_dataset(data_dir, output_dir, test_size=0.2, validation_size=0.1):
    """
    Prepare CINIC-10 dataset for training by splitting into train/validation/test sets.

    Args:
        data_dir (str): Path to raw CINIC-10 dataset
        output_dir (str): Output directory for prepared dataset
        test_size (float): Proportion of data to use for testing
        validation_size (float): Proportion of training data to use for validation

    Returns:
        None
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "valid"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # For demonstration, we'll create a simplified version of the preprocessing
    print(f"Preparing CINIC-10 dataset from {data_dir}")
    print("Note: In a real implementation, you would load the actual CINIC-10 data")
    print("This function demonstrates how the preprocessing would be structured")


def subsample_dataset(source_dir, dest_dir, fraction=None, n_per_class=None):
    """
    Copy a random subset of images from source_dir to dest_dir,
    preserving class subdirectory structure.

    Args:
        source_dir (str): Path to dataset with class subdirectories
        dest_dir (str): Destination path (created if not exists)
        fraction (float): Proportion of images to copy per class (0.0–1.0)
        n_per_class (int): Exact number of images to copy per class

    Raises:
        ValueError: If neither fraction nor n_per_class is provided
    """
    import shutil

    if fraction is None and n_per_class is None:
        raise ValueError("Must specify either fraction or n_per_class")

    for class_name in os.listdir(source_dir):
        class_src = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_src):
            continue

        class_dst = os.path.join(dest_dir, class_name)
        os.makedirs(class_dst, exist_ok=True)

        all_files = [
            f for f in os.listdir(class_src)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if n_per_class is not None:
            n = min(n_per_class, len(all_files))
        else:
            n = max(1, int(len(all_files) * fraction))

        selected = np.random.choice(all_files, size=n, replace=False)

        for fname in selected:
            shutil.copy2(
                os.path.join(class_src, fname),
                os.path.join(class_dst, fname)
            )


# Example usage and testing
if __name__ == "__main__":
    # This is just for demonstration - in actual usage you'd have the real data
    print("Data preprocessing module loaded successfully")
    print("Available functions:")
    print("- load_cinic_data()")
    print("- create_data_generators()")
    print("- get_cinic_statistics()")
    print("- visualize_class_distribution()")
    print("- prepare_cinic_dataset()")
