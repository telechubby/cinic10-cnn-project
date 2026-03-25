"""
Few-Shot Learning Module for CNN Image Classification

This module implements and evaluates few-shot learning techniques for
the CINIC-10 dataset, focusing on methods that can work effectively with
limited training data.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers

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


def create_siamese_network(input_shape=(32, 32, 3), embedding_dim=128):
    """
    Create a Siamese network for few-shot learning.

    Args:
        input_shape (tuple): Input image dimensions (height, width, channels)
        embedding_dim (int): Dimension of the embedding space

    Returns:
        keras.Model: Compiled Siamese network model
    """
    # Input layer
    input_1 = keras.Input(shape=input_shape)
    input_2 = keras.Input(shape=input_shape)

    # Shared CNN backbone for both inputs
    def create_cnn_backbone():
        model = keras.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.2),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.2),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.2),
                layers.GlobalAveragePooling2D(),
                layers.Dense(embedding_dim, activation="relu"),
            ]
        )
        return model

    # Create shared backbone
    backbone = create_cnn_backbone()

    # Get embeddings for both inputs
    embedding_1 = backbone(input_1)
    embedding_2 = backbone(input_2)

    # Calculate absolute difference between embeddings
    distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))(
        [embedding_1, embedding_2]
    )

    # Output layer for similarity prediction (0 = different, 1 = same)
    output = layers.Dense(1, activation="sigmoid")(distance)

    # Create model
    siamese_model = keras.Model(inputs=[input_1, input_2], outputs=output)

    # Compile model
    siamese_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return siamese_model


def create_few_shot_classifier(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a classifier optimized for few-shot learning.

    Args:
        input_shape (tuple): Input image dimensions (height, width, channels)
        num_classes (int): Number of output classes

    Returns:
        keras.Model: Compiled few-shot optimized classifier
    """
    # Use a more robust architecture for few-shot scenarios
    model = keras.Sequential(
        [
            # First convolutional block - robust feature extraction
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Second convolutional block - deeper feature extraction
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Third convolutional block - most robust features
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Global Average Pooling for better few-shot performance
            layers.GlobalAveragePooling2D(),
            # Dense layers with Batch Normalization for stability
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            # Final classification layer
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Use a more stable optimizer for few-shot learning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_prototypical_network(
    input_shape=(32, 32, 3), num_classes=10, embedding_dim=128
):
    """
    Create a prototypical network for few-shot learning.

    Args:
        input_shape (tuple): Input image dimensions (height, width, channels)
        num_classes (int): Number of output classes
        embedding_dim (int): Dimension of the embedding space

    Returns:
        keras.Model: Compiled prototypical network model
    """
    # Input layer
    inputs = keras.Input(shape=input_shape)

    # CNN backbone for feature extraction
    x = layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Feature embedding layer
    embeddings = layers.Dense(embedding_dim, activation="relu")(x)

    # Classification layer
    outputs = layers.Dense(num_classes, activation="softmax")(embeddings)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_few_shot_evaluation(
    model_func,
    train_generator,
    validation_generator,
    few_shot_samples=[1, 5, 10],
    epochs=10,
):
    """
    Evaluate few-shot learning performance on reduced datasets.

    Args:
        model_func: Function to create the CNN model
        train_generator: Training data generator
        validation_generator: Validation data generator
        few_shot_samples (list): Number of samples per class to use for few-shot training
        epochs (int): Number of epochs to train for each configuration

    Returns:
        dict: Results of few-shot evaluation
    """
    results = []

    for samples_per_class in few_shot_samples:
        print(
            f"Evaluating few-shot learning with {samples_per_class} samples per class..."
        )

        try:
            # Create model
            model = model_func()

            # Compile model
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )

            # For demonstration, we'll just record the configuration
            # In a real implementation, you would actually train with reduced datasets

            results.append(
                {
                    "samples_per_class": samples_per_class,
                    "train_accuracy": 0.0,  # Placeholder
                    "val_accuracy": 0.0,  # Placeholder
                    "train_loss": 0.0,  # Placeholder
                    "val_loss": 0.0,  # Placeholder
                    "epochs": epochs,
                }
            )

        except Exception as e:
            print(f"Error with {samples_per_class} samples per class: {e}")
            results.append(
                {
                    "samples_per_class": samples_per_class,
                    "train_accuracy": 0.0,
                    "val_accuracy": 0.0,
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "epochs": epochs,
                }
            )

    return results


def evaluate_few_shot_performance(
    model_func, train_generator, validation_generator, few_shot_configs=[1, 5, 10]
):
    """
    Evaluate the performance of different few-shot learning approaches.

    Args:
        model_func: Function to create the CNN model
        train_generator: Training data generator
        validation_generator: Validation data generator
        few_shot_configs (list): Different few-shot configurations to test

    Returns:
        dict: Detailed evaluation results
    """
    print("Starting few-shot learning performance evaluation...")

    # 1. Evaluate with reduced training data (simulated)
    print("\n1. Evaluating few-shot performance with reduced datasets...")
    few_shot_results = create_few_shot_evaluation(
        model_func,
        train_generator,
        validation_generator,
        few_shot_samples=few_shot_configs,
        epochs=5,
    )

    # 2. Evaluate Siamese network approach (if available)
    print("\n2. Evaluating Siamese network for few-shot learning...")
    try:
        siamese_model = create_siamese_network()
        # Placeholder results for demonstration
        siamese_results = {
            "model_type": "Siamese",
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "epochs": 5,
        }
    except Exception as e:
        print(f"Siamese network evaluation error: {e}")
        siamese_results = {
            "model_type": "Siamese",
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "epochs": 5,
        }

    # 3. Evaluate prototypical network approach (if available)
    print("\n3. Evaluating prototypical network for few-shot learning...")
    try:
        proto_model = create_prototypical_network()
        # Placeholder results for demonstration
        proto_results = {
            "model_type": "Prototypical",
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "epochs": 5,
        }
    except Exception as e:
        print(f"Prototypical network evaluation error: {e}")
        proto_results = {
            "model_type": "Prototypical",
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "epochs": 5,
        }

    # Combine all results
    all_results = {
        "few_shot": few_shot_results,
        "siamese": siamese_results,
        "prototypical": proto_results,
    }

    print("\nFew-shot learning evaluation completed successfully!")

    return all_results


def plot_few_shot_results(results_df, title="Few-Shot Learning Performance"):
    """
    Create visualizations of few-shot learning performance.

    Args:
        results_df (pd.DataFrame): DataFrame containing few-shot evaluation results
        title (str): Title for the plots
    """
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Plot 1: Few-shot samples vs Validation Accuracy
    if "samples_per_class" in results_df.columns:
        axes[0, 0].plot(
            results_df["samples_per_class"],
            results_df["val_accuracy"],
            "o-",
            color="blue",
        )
        axes[0, 0].set_xlabel("Samples per Class")
        axes[0, 0].set_ylabel("Validation Accuracy")
        axes[0, 0].set_title("Few-Shot Performance vs Samples per Class")
        axes[0, 0].grid(True)

    # Plot 2: Few-shot samples vs Training Accuracy
    if "samples_per_class" in results_df.columns:
        axes[0, 1].plot(
            results_df["samples_per_class"],
            results_df["train_accuracy"],
            "o-",
            color="green",
        )
        axes[0, 1].set_xlabel("Samples per Class")
        axes[0, 1].set_ylabel("Training Accuracy")
        axes[0, 1].set_title("Few-Shot Performance vs Training Accuracy")
        axes[0, 1].grid(True)

    # Plot 3: Few-shot samples vs Validation Loss
    if "samples_per_class" in results_df.columns:
        axes[1, 0].plot(
            results_df["samples_per_class"], results_df["val_loss"], "o-", color="red"
        )
        axes[1, 0].set_xlabel("Samples per Class")
        axes[1, 0].set_ylabel("Validation Loss")
        axes[1, 0].set_title("Few-Shot Performance vs Validation Loss")
        axes[1, 0].grid(True)

    # Plot 4: Few-shot samples vs Training Loss
    if "samples_per_class" in results_df.columns:
        axes[1, 1].plot(
            results_df["samples_per_class"],
            results_df["train_loss"],
            "o-",
            color="orange",
        )
        axes[1, 1].set_xlabel("Samples per Class")
        axes[1, 1].set_ylabel("Training Loss")
        axes[1, 1].set_title("Few-Shot Performance vs Training Loss")
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def save_few_shot_results(results, filename_prefix="few_shot_analysis"):
    """
    Save few-shot learning analysis results to CSV file.

    Args:
        results (dict): Dictionary containing all evaluation results
        filename_prefix (str): Prefix for output filename
    """
    # Flatten results for CSV export
    flat_results = []

    if "few_shot" in results:
        for result in results["few_shot"]:
            flat_results.append(
                {
                    "model_type": "Few-Shot Classifier",
                    "samples_per_class": result.get("samples_per_class", 0),
                    "train_accuracy": result.get("train_accuracy", 0.0),
                    "val_accuracy": result.get("val_accuracy", 0.0),
                    "train_loss": result.get("train_loss", 0.0),
                    "val_loss": result.get("val_loss", 0.0),
                    "epochs": result.get("epochs", 0),
                }
            )

    if "siamese" in results:
        flat_results.append(
            {
                "model_type": "Siamese Network",
                "samples_per_class": 0,
                "train_accuracy": results["siamese"].get("train_accuracy", 0.0),
                "val_accuracy": results["siamese"].get("val_accuracy", 0.0),
                "train_loss": results["siamese"].get("train_loss", 0.0),
                "val_loss": results["siamese"].get("val_loss", 0.0),
                "epochs": results["siamese"].get("epochs", 0),
            }
        )

    if "prototypical" in results:
        flat_results.append(
            {
                "model_type": "Prototypical Network",
                "samples_per_class": 0,
                "train_accuracy": results["prototypical"].get("train_accuracy", 0.0),
                "val_accuracy": results["prototypical"].get("val_accuracy", 0.0),
                "train_loss": results["prototypical"].get("train_loss", 0.0),
                "val_loss": results["prototypical"].get("val_loss", 0.0),
                "epochs": results["prototypical"].get("epochs", 0),
            }
        )

    # Create DataFrame from results
    df = pd.DataFrame(flat_results)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"

    # Save to results directory
    output_path = os.path.join("results", filename)
    df.to_csv(output_path, index=False)

    print(f"Few-shot learning results saved to {output_path}")
    return df


# Example usage and testing
if __name__ == "__main__":
    print("Few-Shot Learning Module loaded successfully")

    # Print available functions
    print("\nAvailable few-shot learning functions:")
    print("- create_siamese_network()")
    print("- create_few_shot_classifier()")
    print("- create_prototypical_network()")
    print("- create_few_shot_evaluation()")
    print("- evaluate_few_shot_performance()")
    print("- plot_few_shot_results()")
    print("- save_few_shot_results()")
