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
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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
        subset (str): Which subset to load ('train', 'validation', 'test')

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
    Create data generators for training and validation with optional augmentation.

    Args:
        train_dir (str): Path to training data directory
        validation_dir (str): Path to validation data directory
        batch_size (int): Batch size for generators
        augment (bool): Whether to apply data augmentation

    Returns:
        tuple: (train_generator, validation_generator)
    """

    if augment:
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
        )
    else:
        # No augmentation for training
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Validation data generator (no augmentation)
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, validation_generator


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
