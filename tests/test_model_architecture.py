# tests/test_model_architecture.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import pytest


def _forward(model):
    """Run a single forward pass with a (1,3,32,32) zero tensor."""
    model.train(mode=False)
    with torch.no_grad():
        return model(torch.zeros(1, 3, 32, 32))


def test_create_baseline_cnn_output_shape():
    from model_architecture import create_baseline_cnn
    assert _forward(create_baseline_cnn()).shape == (1, 10)


def test_create_deep_cnn_output_shape():
    from model_architecture import create_deep_cnn
    assert _forward(create_deep_cnn()).shape == (1, 10)


def test_create_efficient_cnn_output_shape():
    from model_architecture import create_efficient_cnn
    assert _forward(create_efficient_cnn()).shape == (1, 10)


def test_create_cnn_with_regularization_output_shape():
    from model_architecture import create_cnn_with_regularization
    assert _forward(create_cnn_with_regularization()).shape == (1, 10)


def test_create_few_shot_cnn_output_shape():
    from model_architecture import create_few_shot_cnn
    assert _forward(create_few_shot_cnn()).shape == (1, 10)


def test_models_are_nn_modules():
    from model_architecture import (
        create_baseline_cnn, create_deep_cnn, create_efficient_cnn,
        create_cnn_with_regularization, create_few_shot_cnn,
    )
    for fn in [create_baseline_cnn, create_deep_cnn, create_efficient_cnn,
               create_cnn_with_regularization, create_few_shot_cnn]:
        assert isinstance(fn(), nn.Module)


def test_create_cnn_with_regularization_accepts_args():
    from model_architecture import create_cnn_with_regularization
    out = _forward(create_cnn_with_regularization(dropout_rate=0.1, weight_decay=1e-3))
    assert out.shape == (1, 10)


def test_no_nan_in_output():
    from model_architecture import create_baseline_cnn
    import torch
    torch.manual_seed(0)
    model = create_baseline_cnn()
    model.train(mode=False)
    with torch.no_grad():
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
    assert not torch.isnan(out).any(), "NaN detected in model output"
    assert not torch.isinf(out).any(), "Inf detected in model output"
