"""
Hyperparameter Analysis Module for CNN Training

This module conducts comprehensive analysis of various hyperparameters
that affect CNN performance on the CINIC-10 dataset, including learning rates,
batch sizes, and regularization parameters.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set random seeds for reproducibility (numpy-only at module level)
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


def analyze_learning_rates(
    model_func, train_dir, val_dir,
    learning_rates=None, epochs=10, batch_size=32
):
    """
    Analyze the impact of different learning rates on model performance.

    Args:
        model_func: Function to create the CNN model
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory
        learning_rates (list): List of learning rates to test
        epochs (int): Number of epochs to train for each rate
        batch_size (int): Batch size for training

    Returns:
        list: Results of learning rate analysis
    """
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if learning_rates is None:
        learning_rates = [0.0001, 0.001, 0.01, 0.1]

    results = []
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=(32, 32), batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )

    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_gen = train_datagen.flow_from_directory(
            train_dir, target_size=(32, 32), batch_size=batch_size,
            class_mode="categorical", shuffle=True
        )
        model = model_func()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy", metrics=["accuracy"]
        )
        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=0)
        results.append({
            "learning_rate": lr,
            "train_accuracy": history.history["accuracy"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
            "train_loss": history.history["loss"][-1],
            "val_loss": history.history["val_loss"][-1],
            "epochs": epochs,
        })
    return results


def analyze_batch_sizes(
    model_func, train_dir, val_dir,
    batch_sizes=None, epochs=10
):
    """
    Analyze the impact of different batch sizes on model performance.

    Args:
        model_func: Function to create the CNN model
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory
        batch_sizes (list): List of batch sizes to test
        epochs (int): Number of epochs to train for each size

    Returns:
        list: Results of batch size analysis
    """
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if batch_sizes is None:
        batch_sizes = [32, 64, 128]

    results = []

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_gen = train_datagen.flow_from_directory(
            train_dir, target_size=(32, 32), batch_size=batch_size,
            class_mode="categorical", shuffle=True
        )
        val_gen = val_datagen.flow_from_directory(
            val_dir, target_size=(32, 32), batch_size=batch_size,
            class_mode="categorical", shuffle=False
        )
        model = model_func()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy", metrics=["accuracy"]
        )
        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=0)
        results.append({
            "batch_size": batch_size,
            "train_accuracy": history.history["accuracy"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
            "train_loss": history.history["loss"][-1],
            "val_loss": history.history["val_loss"][-1],
            "epochs": epochs,
        })
    return results


def analyze_regularization_strengths(
    model_func, train_dir, val_dir,
    dropout_rates=None, weight_decays=None,
    epochs=10, batch_size=32
):
    """
    Analyze the impact of regularization techniques on model performance.

    Args:
        model_func: Function to create the CNN model with regularization
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory
        dropout_rates (list): List of dropout rates to test
        weight_decays (list): List of weight decay values to test
        epochs (int): Number of epochs to train for each configuration
        batch_size (int): Batch size for training

    Returns:
        list: Results of regularization analysis
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if dropout_rates is None:
        dropout_rates = [0.2, 0.3, 0.5]
    if weight_decays is None:
        weight_decays = [1e-4, 1e-3, 1e-2]

    results = []
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=(32, 32), batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )

    for dropout_rate in dropout_rates:
        for weight_decay in weight_decays:
            print(f"Testing regularization — Dropout: {dropout_rate}, Weight decay: {weight_decay}")
            train_datagen = ImageDataGenerator(rescale=1.0 / 255)
            train_gen = train_datagen.flow_from_directory(
                train_dir, target_size=(32, 32), batch_size=batch_size,
                class_mode="categorical", shuffle=True
            )
            try:
                model = model_func(dropout_rate=dropout_rate, weight_decay=weight_decay)
                history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=0)
                results.append({
                    "dropout_rate": dropout_rate,
                    "weight_decay": weight_decay,
                    "train_accuracy": history.history["accuracy"][-1],
                    "val_accuracy": history.history["val_accuracy"][-1],
                    "train_loss": history.history["loss"][-1],
                    "val_loss": history.history["val_loss"][-1],
                    "epochs": epochs,
                })
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    "dropout_rate": dropout_rate, "weight_decay": weight_decay,
                    "train_accuracy": 0.0, "val_accuracy": 0.0,
                    "train_loss": 0.0, "val_loss": 0.0, "epochs": epochs,
                })
    return results


def analyze_optimizers(
    model_func, train_dir, val_dir,
    optimizers=None, epochs=10, batch_size=32
):
    """
    Analyze the impact of different optimizers on model performance.

    Args:
        model_func: Function to create the CNN model
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory
        optimizers (list): List of optimizer names to test
        epochs (int): Number of epochs to train for each optimizer
        batch_size (int): Batch size for training

    Returns:
        list: Results of optimizer analysis
    """
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if optimizers is None:
        optimizers = ["adam", "sgd", "rmsprop"]

    results = []
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=(32, 32), batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )

    for optimizer_name in optimizers:
        print(f"Testing optimizer: {optimizer_name}")
        if optimizer_name == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
        elif optimizer_name == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        elif optimizer_name == "rmsprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=0.001)

        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_gen = train_datagen.flow_from_directory(
            train_dir, target_size=(32, 32), batch_size=batch_size,
            class_mode="categorical", shuffle=True
        )
        model = model_func()
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        try:
            history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=0)
            results.append({
                "optimizer": optimizer_name,
                "train_accuracy": history.history["accuracy"][-1],
                "val_accuracy": history.history["val_accuracy"][-1],
                "train_loss": history.history["loss"][-1],
                "val_loss": history.history["val_loss"][-1],
                "epochs": epochs,
            })
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "optimizer": optimizer_name,
                "train_accuracy": 0.0, "val_accuracy": 0.0,
                "train_loss": 0.0, "val_loss": 0.0, "epochs": epochs,
            })
    return results


def plot_hyperparameter_results(results_df, title="Hyperparameter Analysis Results"):
    """
    Create visualizations of hyperparameter analysis results.

    Args:
        results_df (pd.DataFrame): DataFrame containing analysis results
        title (str): Title for the plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Plot 1: Learning rate vs Accuracy
    if "learning_rate" in results_df.columns:
        axes[0, 0].plot(results_df["learning_rate"], results_df["val_accuracy"], "o-")
        axes[0, 0].set_xlabel("Learning Rate")
        axes[0, 0].set_ylabel("Validation Accuracy")
        axes[0, 0].set_title("Learning Rate vs Validation Accuracy")
        axes[0, 0].set_xscale("log")

    # Plot 2: Batch size vs Accuracy (if available)
    if "batch_size" in results_df.columns:
        axes[0, 1].bar(results_df["batch_size"], results_df["val_accuracy"])
        axes[0, 1].set_xlabel("Batch Size")
        axes[0, 1].set_ylabel("Validation Accuracy")
        axes[0, 1].set_title("Batch Size vs Validation Accuracy")

    # Plot 3: Dropout rate vs Accuracy (if available)
    if "dropout_rate" in results_df.columns:
        axes[1, 0].plot(results_df["dropout_rate"], results_df["val_accuracy"], "o-")
        axes[1, 0].set_xlabel("Dropout Rate")
        axes[1, 0].set_ylabel("Validation Accuracy")
        axes[1, 0].set_title("Dropout Rate vs Validation Accuracy")

    # Plot 4: Weight decay vs Accuracy (if available)
    if "weight_decay" in results_df.columns:
        axes[1, 1].plot(results_df["weight_decay"], results_df["val_accuracy"], "o-")
        axes[1, 1].set_xlabel("Weight Decay")
        axes[1, 1].set_ylabel("Validation Accuracy")
        axes[1, 1].set_title("Weight Decay vs Validation Accuracy")
        axes[1, 1].set_xscale("log")

    plt.tight_layout()
    plt.show()


def save_hyperparameter_results(results, filename_prefix="hyperparameter_analysis"):
    """
    Save hyperparameter analysis results to CSV file.

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


def create_comprehensive_hyperparameter_analysis(model_func, train_dir, val_dir):
    """
    Perform comprehensive hyperparameter analysis.

    Args:
        model_func: Function to create the CNN model
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory

    Returns:
        dict: All analysis results
    """
    print("Starting comprehensive hyperparameter analysis...")

    print("\n1. Analyzing learning rates...")
    lr_results = analyze_learning_rates(
        model_func, train_dir, val_dir,
        learning_rates=[0.0001, 0.001, 0.01, 0.1], epochs=5
    )

    print("\n2. Analyzing batch sizes...")
    batch_results = analyze_batch_sizes(
        model_func, train_dir, val_dir,
        batch_sizes=[16, 32, 64], epochs=5
    )

    print("\n3. Analyzing regularization parameters...")
    reg_results = analyze_regularization_strengths(
        model_func, train_dir, val_dir,
        dropout_rates=[0.1, 0.2, 0.3, 0.5],
        weight_decays=[1e-4, 1e-3, 1e-2], epochs=5
    )

    print("\n4. Analyzing optimizers...")
    opt_results = analyze_optimizers(
        model_func, train_dir, val_dir,
        optimizers=["adam", "sgd", "rmsprop"], epochs=5
    )

    print("\nHyperparameter analysis completed successfully!")
    return {
        "learning_rate": lr_results,
        "batch_size": batch_results,
        "regularization": reg_results,
        "optimizer": opt_results,
    }


# Example usage and testing
if __name__ == "__main__":
    print("Hyperparameter Analysis Module loaded successfully")

    # Print available functions
    print("\nAvailable analysis functions:")
    print("- analyze_learning_rates()")
    print("- analyze_batch_sizes()")
    print("- analyze_regularization_strengths()")
    print("- analyze_optimizers()")
    print("- plot_hyperparameter_results()")
    print("- save_hyperparameter_results()")
    print("- create_comprehensive_hyperparameter_analysis()")
