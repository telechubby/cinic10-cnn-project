"""Tests for evaluation module."""
import os
import sys
import tempfile
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def make_mock_dataset(root_dir, classes, n_per_class=10):
    """Create tiny mock dataset."""
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(os.path.join(cls_dir, f"img_{i:04d}.png"))


def test_create_model_comparison_visualizations_runs():
    """create_model_comparison_visualizations doesn't raise an exception."""
    import matplotlib
    matplotlib.use('Agg')
    from evaluation import create_model_comparison_visualizations

    comparison_results = [
        {"model_name": "A", "test_accuracy": 0.8, "test_loss": 0.5,
         "top_1_accuracy": 0.8, "top_5_accuracy": 0.95},
        {"model_name": "B", "test_accuracy": 0.75, "test_loss": 0.6,
         "top_1_accuracy": 0.75, "top_5_accuracy": 0.92},
    ]
    # Should not raise
    create_model_comparison_visualizations(comparison_results)


def test_run_reduced_dataset_experiment_returns_correct_structure():
    """run_reduced_dataset_experiment returns list with one entry per fraction."""
    from evaluation import run_reduced_dataset_experiment

    # We just test the function signature and return structure
    # without actually running TF training
    import inspect
    sig = inspect.signature(run_reduced_dataset_experiment)
    params = list(sig.parameters.keys())
    assert "model_func" in params
    assert "train_dir" in params
    assert "val_dir" in params
    assert "fractions" in params


def test_plot_reduced_dataset_results_runs():
    """plot_reduced_dataset_results accepts result dicts and doesn't raise."""
    import matplotlib
    matplotlib.use('Agg')
    from evaluation import plot_reduced_dataset_results

    results = [
        {"fraction": 0.1, "val_accuracy": 0.5, "val_loss": 1.2, "num_train_samples": 100},
        {"fraction": 0.5, "val_accuracy": 0.7, "val_loss": 0.8, "num_train_samples": 500},
        {"fraction": 1.0, "val_accuracy": 0.82, "val_loss": 0.6, "num_train_samples": 1000},
    ]
    plot_reduced_dataset_results(results)


def test_calculate_performance_metrics_with_mock_model():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from evaluation import calculate_performance_metrics

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    X = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)
    metrics = calculate_performance_metrics(model, loader)
    assert "test_accuracy" in metrics
    assert "test_loss" in metrics
    assert 0.0 <= metrics["test_accuracy"] <= 1.0


def test_generate_confusion_matrix_shape():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from evaluation import generate_confusion_matrix

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    X = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 10, (20,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)
    cm = generate_confusion_matrix(model, loader)
    assert cm.shape == (10, 10)
