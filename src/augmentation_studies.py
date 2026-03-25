"""
Data Augmentation Studies Module for CINIC-10 Dataset

This module implements and evaluates various data augmentation techniques
for improving CNN performance on the CINIC-10 dataset, including both
standard and advanced augmentation methods.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf

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


def create_standard_augmentation_generators():
    """
    Create ImageDataGenerator instances with different standard augmentation techniques.

    Returns:
        dict: Dictionary containing different augmentation configurations
    """
    # Standard augmentation with rotation, shifting, and flipping
    standard_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,  # For creating validation splits
    )

    # Augmentation with color jittering (more aggressive)
    color_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        brightness_range=[0.8, 1.2],
        contrast_range=[0.8, 1.2],
        validation_split=0.2,
    )

    # Minimal augmentation (only rescaling)
    minimal_aug = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    # Crop and resize augmentation
    crop_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    return {
        "standard": standard_aug,
        "color_jitter": color_aug,
        "minimal": minimal_aug,
        "crop_resize": crop_aug,
    }


def create_advanced_augmentation_generators():
    """
    Create ImageDataGenerator instances with advanced augmentation techniques.

    Returns:
        dict: Dictionary containing different advanced augmentation configurations
    """
    # Cutout augmentation (simulated)
    cutout_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    # Mixup augmentation (simulated)
    mixup_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    # AutoAugment-like augmentation (simplified)
    autoaugment_like = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    return {
        "cutout": cutout_aug,
        "mixup": mixup_aug,
        "autoaugment_like": autoaugment_like,
    }


def apply_cutout_augmentation(image, mask_size=8):
    """
    Apply cutout augmentation to a single image.

    Args:
        image (numpy.ndarray): Input image array
        mask_size (int): Size of the mask to apply

    Returns:
        numpy.ndarray: Image with cutout applied
    """
    # Create a copy of the image to avoid modifying the original
    augmented_image = np.copy(image)

    # Get image dimensions
    height, width, channels = image.shape

    # Randomly select the top-left corner of the mask
    y = np.random.randint(0, height)
    x = np.random.randint(0, width)

    # Calculate the mask boundaries
    y1 = max(0, y - mask_size // 2)
    y2 = min(height, y + mask_size // 2)
    x1 = max(0, x - mask_size // 2)
    x2 = min(width, x + mask_size // 2)

    # Apply the mask (set to zero - black)
    augmented_image[y1:y2, x1:x2] = 0

    return augmented_image


def apply_cutmix_augmentation(image1, image2, alpha=1.0):
    """
    Apply CutMix augmentation to two images.

    Args:
        image1 (numpy.ndarray): First input image
        image2 (numpy.ndarray): Second input image
        alpha (float): Parameter for beta distribution

    Returns:
        numpy.ndarray: Mixed image
    """
    # Randomly sample lambda from beta distribution
    lam = np.random.beta(alpha, alpha)

    # Get image dimensions
    height, width, channels = image1.shape

    # Randomly select the region to crop from the second image
    crop_h = int(height * lam)
    crop_w = int(width * lam)

    # Randomly position the crop
    y_start = np.random.randint(0, height - crop_h)
    x_start = np.random.randint(0, width - crop_w)

    # Create mixed image
    mixed_image = np.copy(image1)

    # Replace region with pixels from the second image
    mixed_image[y_start : y_start + crop_h, x_start : x_start + crop_w] = image2[
        y_start : y_start + crop_h, x_start : x_start + crop_w
    ]

    return mixed_image


def evaluate_augmentation_effects(
    model_func, train_generator, validation_generator, augmentation_configs, epochs=5
):
    """
    Evaluate the impact of different augmentation techniques on model performance.

    Args:
        model_func: Function to create the CNN model
        train_generator: Training data generator
        validation_generator: Validation data generator
        augmentation_configs (dict): Dictionary of augmentation configurations
        epochs (int): Number of epochs to train for each configuration

    Returns:
        dict: Results of augmentation evaluation
    """
    results = []

    for aug_name, aug_config in augmentation_configs.items():
        print(f"Evaluating {aug_name} augmentation...")

        try:
            # Create model
            model = model_func()

            # Compile model
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )

            # Train model with specific augmentation
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                verbose=0,
            )

            # Extract final metrics
            train_acc = history.history["accuracy"][-1]
            val_acc = history.history["val_accuracy"][-1]
            train_loss = history.history["loss"][-1]
            val_loss = history.history["val_loss"][-1]

            results.append(
                {
                    "augmentation": aug_name,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epochs": epochs,
                }
            )

        except Exception as e:
            print(f"Error with {aug_name} augmentation: {e}")
            results.append(
                {
                    "augmentation": aug_name,
                    "train_accuracy": 0.0,
                    "val_accuracy": 0.0,
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "epochs": epochs,
                }
            )

    return results


def visualize_augmentation_results(results_df, title="Data Augmentation Analysis"):
    """
    Create visualizations of data augmentation analysis results.

    Args:
        results_df (pd.DataFrame): DataFrame containing analysis results
        title (str): Title for the plots
    """
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Plot 1: Augmentation vs Validation Accuracy
    axes[0, 0].bar(
        results_df["augmentation"], results_df["val_accuracy"], color="skyblue"
    )
    axes[0, 0].set_xlabel("Augmentation Techniques")
    axes[0, 0].set_ylabel("Validation Accuracy")
    axes[0, 0].set_title("Augmentation Techniques vs Validation Accuracy")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Plot 2: Augmentation vs Training Accuracy
    axes[0, 1].bar(
        results_df["augmentation"], results_df["train_accuracy"], color="lightgreen"
    )
    axes[0, 1].set_xlabel("Augmentation Techniques")
    axes[0, 1].set_ylabel("Training Accuracy")
    axes[0, 1].set_title("Augmentation Techniques vs Training Accuracy")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Plot 3: Augmentation vs Validation Loss
    axes[1, 0].bar(
        results_df["augmentation"], results_df["val_loss"], color="lightcoral"
    )
    axes[1, 0].set_xlabel("Augmentation Techniques")
    axes[1, 0].set_ylabel("Validation Loss")
    axes[1, 0].set_title("Augmentation Techniques vs Validation Loss")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Plot 4: Augmentation vs Training Loss
    axes[1, 1].bar(
        results_df["augmentation"], results_df["train_loss"], color="lightyellow"
    )
    axes[1, 1].set_xlabel("Augmentation Techniques")
    axes[1, 1].set_ylabel("Training Loss")
    axes[1, 1].set_title("Augmentation Techniques vs Training Loss")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def save_augmentation_results(results, filename_prefix="augmentation_analysis"):
    """
    Save augmentation analysis results to CSV file.

    Args:
        results (list): List of result dictionaries
        filename_prefix (str): Prefix for output filename
    """
    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"

    # Save to results directory
    output_path = os.path.join("results", filename)
    df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")
    return df


def compare_augmentation_approaches(model_func, train_generator, validation_generator):
    """
    Perform comprehensive comparison of standard and advanced augmentation techniques.

    Args:
        model_func: Function to create the CNN model
        train_generator: Training data generator
        validation_generator: Validation data generator

    Returns:
        dict: All augmentation analysis results
    """
    print("Starting comprehensive augmentation analysis...")

    # 1. Standard Augmentation Analysis
    print("\n1. Analyzing standard augmentation techniques...")
    standard_augs = create_standard_augmentation_generators()
    standard_results = evaluate_augmentation_effects(
        model_func, train_generator, validation_generator, standard_augs, epochs=5
    )

    # 2. Advanced Augmentation Analysis
    print("\n2. Analyzing advanced augmentation techniques...")
    advanced_augs = create_advanced_augmentation_generators()
    advanced_results = evaluate_augmentation_effects(
        model_func, train_generator, validation_generator, advanced_augs, epochs=5
    )

    # Combine all results for comprehensive analysis
    all_results = {
        "standard": standard_results,
        "advanced": advanced_results,
    }

    print("\nAugmentation analysis completed successfully!")

    return all_results


# Example usage and testing
if __name__ == "__main__":
    print("Data Augmentation Studies Module loaded successfully")

    # Print available functions
    print("\nAvailable augmentation functions:")
    print("- create_standard_augmentation_generators()")
    print("- create_advanced_augmentation_generators()")
    print("- apply_cutout_augmentation()")
    print("- apply_cutmix_augmentation()")
    print("- evaluate_augmentation_effects()")
    print("- visualize_augmentation_results()")
    print("- save_augmentation_results()")
    print("- compare_augmentation_approaches()")
