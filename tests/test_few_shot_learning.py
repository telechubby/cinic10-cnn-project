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
    """forward(x1, x2) returns (B, 1) sigmoid similarity."""
    from few_shot_learning import create_siamese_network
    model = create_siamese_network()
    model.train(mode=False)
    x1, x2 = torch.zeros(2, 3, 32, 32), torch.zeros(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x1, x2)
    assert out.shape == (2, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_siamese_shared_backbone():
    from few_shot_learning import create_siamese_network
    model = create_siamese_network()
    assert hasattr(model, "backbone")


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
