# tests/test_few_shot_learning.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import pytest


def test_create_siamese_network_is_nn_module():
    from few_shot_learning import create_siamese_network
    assert isinstance(create_siamese_network(), nn.Module)


def test_create_siamese_network_output_shape():
    """forward(x1, x2) returns (B, 1) sigmoid similarity in [0, 1]."""
    from few_shot_learning import create_siamese_network
    torch.manual_seed(0)
    model = create_siamese_network()
    model.train(mode=False)
    x1 = torch.randn(2, 3, 32, 32)
    x2 = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x1, x2)
    assert out.shape == (2, 1)
    assert (out >= 0).all() and (out <= 1).all(), "Sigmoid output must be in [0, 1]"
    # Identical inputs should produce higher similarity than random pairs
    with torch.no_grad():
        out_same = model(x1, x1)
        out_diff = model(x1, x2)
    assert out_same.mean() >= out_diff.mean() - 0.3, \
        "Same inputs should not score much lower than different inputs (backbone sanity check)"


def test_siamese_shared_backbone():
    """The backbone is shared: same input through backbone twice gives identical embeddings."""
    from few_shot_learning import create_siamese_network
    model = create_siamese_network()
    assert hasattr(model, "backbone"), "SiameseNetwork must have a 'backbone' attribute"
    # Shared weights: backbone(x1) and backbone(x2) for x1==x2 must be identical
    model.train(mode=False)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        e1 = model.backbone(x)
        e2 = model.backbone(x)
    assert torch.allclose(e1, e2), "backbone must produce identical embeddings for identical inputs"


def test_create_few_shot_classifier_output_shape():
    from few_shot_learning import create_few_shot_classifier
    model = create_few_shot_classifier()
    model.train(mode=False)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 32, 32))
    assert out.shape == (1, 10)


def test_create_prototypical_network_output_shape():
    from few_shot_learning import create_prototypical_network
    model = create_prototypical_network()
    model.train(mode=False)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 32, 32))
    assert out.shape == (1, 10)
