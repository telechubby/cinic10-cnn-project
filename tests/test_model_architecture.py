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


# ── VGGBaseline tests ─────────────────────────────────────────────────────────

def test_vgg_baseline_output_shape():
    from model_architecture import create_vgg_baseline
    assert _forward(create_vgg_baseline()).shape == (1, 10)


def test_vgg_baseline_is_nn_module():
    from model_architecture import create_vgg_baseline
    assert isinstance(create_vgg_baseline(), nn.Module)


def test_vgg_baseline_no_nan():
    from model_architecture import create_vgg_baseline
    torch.manual_seed(0)
    model = create_vgg_baseline()
    model.train(mode=False)
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ── ResNetDeep tests ──────────────────────────────────────────────────────────

def test_resnet_deep_output_shape():
    from model_architecture import create_resnet_deep
    assert _forward(create_resnet_deep()).shape == (1, 10)


def test_resnet_deep_is_nn_module():
    from model_architecture import create_resnet_deep
    assert isinstance(create_resnet_deep(), nn.Module)


def test_resnet_deep_no_nan():
    from model_architecture import create_resnet_deep
    torch.manual_seed(0)
    model = create_resnet_deep()
    model.train(mode=False)
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_resnet_deep_residual_skip_shapes():
    """Verify residual blocks produce correct spatial dimensions."""
    from model_architecture import ResNetDeep
    torch.manual_seed(0)
    model = ResNetDeep()
    model.train(mode=False)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        s  = model.stem(x)         # (1, 64,  32, 32)
        s1 = model.stage1(s)       # (1, 64,  32, 32)
        s2 = model.stage2(s1)      # (1, 128, 16, 16)
        s3 = model.stage3(s2)      # (1, 256,  8,  8)
    assert s.shape  == (1,  64, 32, 32)
    assert s1.shape == (1,  64, 32, 32)
    assert s2.shape == (1, 128, 16, 16)
    assert s3.shape == (1, 256,  8,  8)
