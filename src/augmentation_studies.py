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

# Set random seeds for reproducibility
np.random.seed(42)

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


class CutoutLayer:
    """
    Keras layer applying Cutout augmentation during training.
    Zeros out a random square patch of each image in the batch.
    Passthrough at inference time.
    """

    def __init__(self, mask_size=8, **kwargs):
        self.mask_size = mask_size
        self.kwargs = kwargs

    def call(self, inputs, training=None):
        import tensorflow as tf
        if not training:
            return inputs

        images = inputs
        batch_size = tf.shape(images)[0]
        h = tf.shape(images)[1]
        w = tf.shape(images)[2]
        half = self.mask_size // 2

        def apply_cutout(img):
            cx = tf.random.uniform((), 0, w, dtype=tf.int32)
            cy = tf.random.uniform((), 0, h, dtype=tf.int32)
            x1 = tf.maximum(0, cx - half)
            x2 = tf.minimum(w, cx + half)
            y1 = tf.maximum(0, cy - half)
            y2 = tf.minimum(h, cy + half)

            # Build binary mask (1 = keep, 0 = zero out)
            rows = tf.range(h)
            cols = tf.range(w)
            row_mask = tf.logical_or(rows < y1, rows >= y2)
            col_mask = tf.logical_or(cols < x1, cols >= x2)
            mask_2d = tf.logical_or(
                tf.reshape(row_mask, [-1, 1]),
                tf.reshape(col_mask, [1, -1])
            )
            mask_3d = tf.cast(tf.expand_dims(mask_2d, -1), img.dtype)
            return img * mask_3d

        return tf.map_fn(apply_cutout, images)

    def get_config(self):
        config = {"mask_size": self.mask_size}
        config.update(self.kwargs)
        return config


def create_standard_augmentation_generators():
    """
    Create ImageDataGenerator instances with different standard augmentation techniques.

    Returns:
        dict: Dictionary containing different augmentation configurations
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    )

    # Minimal augmentation (only rescaling)
    minimal_aug = ImageDataGenerator(rescale=1.0 / 255)

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
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Cutout augmentation (placeholder - will be replaced with actual CutoutLayer)
    cutout_aug = None

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


def create_cutout_tf_dataset(train_dir, batch_size=32, mask_size=8):
    """
    Build a tf.data.Dataset pipeline with CutoutLayer augmentation.

    Args:
        train_dir (str): Path to training data directory (class subdirs)
        batch_size (int): Batch size
        mask_size (int): Cutout mask size in pixels

    Returns:
        tf.data.Dataset: Batched dataset with cutout applied during training
    """
    import tensorflow as tf

    dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(32, 32),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
    )
    cutout = CutoutLayer(mask_size=mask_size)
    normalize = tf.keras.layers.Rescaling(1.0 / 255)

    dataset = dataset.map(
        lambda x, y: (cutout(normalize(x), training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def evaluate_augmentation_effects(
    model_func, train_dir, val_dir, augmentation_configs, epochs=5, batch_size=32
):
    """
    Evaluate the impact of different augmentation techniques on model performance.

    Args:
        model_func: Function to create the CNN model
        train_dir (str): Path to training data directory (class subdirs)
        val_dir (str): Path to validation data directory (class subdirs)
        augmentation_configs (dict): Maps name -> ImageDataGenerator instance OR tf.data.Dataset
        epochs (int): Number of epochs per configuration
        batch_size (int): Batch size for ImageDataGenerator-based configs

    Returns:
        list: Dicts with augmentation, train_accuracy, val_accuracy, train_loss, val_loss
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    results = []

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=(32, 32), batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )

    for aug_name, aug_config in augmentation_configs.items():
        if aug_config is None:
            continue  # skip placeholder entries
        print(f"Evaluating {aug_name} augmentation...")

        try:
            model = model_func()
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )

            if isinstance(aug_config, ImageDataGenerator):
                train_gen = aug_config.flow_from_directory(
                    train_dir, target_size=(32, 32), batch_size=batch_size,
                    class_mode="categorical", shuffle=True
                )
                history = model.fit(
                    train_gen, epochs=epochs, validation_data=val_gen, verbose=0
                )
            else:
                # tf.data.Dataset (e.g. cutout pipeline)
                history = model.fit(
                    aug_config, epochs=epochs, validation_data=val_gen, verbose=0
                )

            results.append({
                "augmentation": aug_name,
                "train_accuracy": history.history["accuracy"][-1],
                "val_accuracy": history.history["val_accuracy"][-1],
                "train_loss": history.history["loss"][-1],
                "val_loss": history.history["val_loss"][-1],
                "epochs": epochs,
            })

        except Exception as e:
            print(f"Error with {aug_name}: {e}")
            results.append({
                "augmentation": aug_name,
                "train_accuracy": 0.0, "val_accuracy": 0.0,
                "train_loss": 0.0, "val_loss": 0.0, "epochs": epochs,
            })

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


def compare_augmentation_approaches(model_func, train_dir, val_dir, epochs=5, batch_size=32):
    """
    Perform comprehensive comparison of standard and advanced augmentation techniques.

    Args:
        model_func: Function to create the CNN model
        train_dir (str): Path to training data (class subdirs)
        val_dir (str): Path to validation data (class subdirs)
        epochs (int): Training epochs per augmentation config
        batch_size (int): Batch size

    Returns:
        dict: {"standard": [...], "advanced": [...]}
    """
    print("Starting comprehensive augmentation analysis...")

    print("\n1. Analyzing standard augmentation techniques...")
    standard_augs = create_standard_augmentation_generators()
    standard_results = evaluate_augmentation_effects(
        model_func, train_dir, val_dir, standard_augs,
        epochs=epochs, batch_size=batch_size
    )

    print("\n2. Analyzing advanced augmentation techniques...")
    advanced_augs = create_advanced_augmentation_generators()
    # Replace placeholder cutout entry with real tf.data pipeline
    advanced_augs["cutout"] = create_cutout_tf_dataset(train_dir, batch_size=batch_size)
    advanced_results = evaluate_augmentation_effects(
        model_func, train_dir, val_dir, advanced_augs,
        epochs=epochs, batch_size=batch_size
    )

    print("\nAugmentation analysis completed successfully!")
    return {"standard": standard_results, "advanced": advanced_results}


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
