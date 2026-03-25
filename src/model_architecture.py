"""
CNN Model Architecture Module for CINIC-10 Image Classification

This module defines various CNN architectures suitable for the CINIC-10 dataset,
including baseline models and specialized architectures for different experimental
requirements.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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


def create_baseline_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a baseline CNN architecture for CINIC-10 classification.

    Args:
        input_shape (tuple): Input image dimensions (height, width, channels)
        num_classes (int): Number of output classes

    Returns:
        keras.Model: Compiled CNN model
    """
    model = keras.Sequential(
        [
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def create_deep_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a deeper CNN architecture for better performance.

    Args:
        input_shape (tuple): Input image dimensions (height, width, channels)
        num_classes (int): Number of output classes

    Returns:
        keras.Model: Compiled CNN model
    """
    model = keras.Sequential(
        [
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def create_efficient_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a more efficient CNN architecture optimized for smaller datasets.

    Args:
        input_shape (tuple): Input image dimensions (height, width, channels)
        num_classes (int): Number of output classes

    Returns:
        keras.Model: Compiled CNN model
    """
    model = keras.Sequential(
        [
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile model with a more conservative learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_cnn_with_regularization(
    input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.3, weight_decay=1e-4
):
    """
    Create CNN with enhanced regularization techniques.

    Args:
        input_shape (tuple): Input image dimensions (height, width, channels)
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        weight_decay (float): L2 regularization strength

    Returns:
        keras.Model: Compiled CNN model with regularization
    """
    # Using Keras functional API for more complex architectures
    inputs = keras.Input(shape=input_shape)

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        padding="same",
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        padding="same",
    )(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        padding="same",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        padding="same",
    )(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Third conv block
    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        padding="same",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        padding="same",
    )(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(
        512, activation="relu", kernel_regularizer=keras.regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate * 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    # Compile with appropriate optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_few_shot_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Create a specialized CNN architecture optimized for few-shot learning.

    Args:
        input_shape (tuple): Input image dimensions (height, width, channels)
        num_classes (int): Number of output classes

    Returns:
        keras.Model: Compiled CNN model optimized for few-shot learning
    """
    # This architecture emphasizes feature extraction capabilities for few-shot scenarios
    model = keras.Sequential(
        [
            # Feature extraction layers - more robust to few samples
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Deeper feature extraction
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Even deeper feature extraction
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            # Global average pooling for better few-shot performance
            layers.GlobalAveragePooling2D(),
            # Dense layers with batch normalization for stability
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
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


def get_model_summary(model, model_name="Model"):
    """
    Print a summary of the model architecture.

    Args:
        model (keras.Model): The compiled Keras model
        model_name (str): Name of the model for display
    """
    print(f"\n{model_name} Architecture Summary:")
    print("=" * 50)
    model.summary()


# Example usage and testing
if __name__ == "__main__":
    print("CNN Model Architecture Module loaded successfully")

    # Test creating different model architectures
    try:
        baseline_model = create_baseline_cnn()
        print("✓ Baseline CNN model created successfully")

        deep_model = create_deep_cnn()
        print("✓ Deep CNN model created successfully")

        efficient_model = create_efficient_cnn()
        print("✓ Efficient CNN model created successfully")

        regularized_model = create_cnn_with_regularization()
        print("✓ Regularized CNN model created successfully")

        few_shot_model = create_few_shot_cnn()
        print("✓ Few-shot CNN model created successfully")

    except Exception as e:
        print(f"Error creating models: {e}")

    print("\nAvailable model creation functions:")
    print("- create_baseline_cnn()")
    print("- create_deep_cnn()")
    print("- create_efficient_cnn()")
    print("- create_cnn_with_regularization()")
    print("- create_few_shot_cnn()")
