# tests/test_utils.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
import pytest
from torch.utils.data import DataLoader, TensorDataset


def test_get_device_returns_torch_device():
    from utils import get_device
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("mps", "cpu")


def test_set_seeds_runs_without_error():
    from utils import set_seeds
    set_seeds(42)  # must not raise


def test_train_model_returns_history_lists():
    from utils import train_model
    device = torch.device("cpu")
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4)).to(device)
    X = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 4, (8,))
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=4)
    optimizer = torch.optim.Adam(model.parameters())
    history = train_model(model, loader, loader, optimizer, epochs=2, device=device)
    assert set(history.keys()) == {"loss", "accuracy", "val_loss", "val_accuracy"}
    assert len(history["loss"]) == 2
    assert len(history["val_accuracy"]) == 2


def test_train_model_early_stopping():
    from utils import train_model
    device = torch.device("cpu")
    # Use a frozen (non-learning) model so val_accuracy never improves,
    # guaranteeing early stopping triggers regardless of random seed.
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4)).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    X = torch.zeros(8, 3, 8, 8)  # constant input -> constant output -> no accuracy gain
    y = torch.zeros(8, dtype=torch.long)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0 => weights never move
    history = train_model(model, loader, loader, optimizer,
                          epochs=10, device=device, patience=1)
    assert len(history["loss"]) < 10


def test_train_model_checkpoint(tmp_path):
    from utils import train_model
    device = torch.device("cpu")
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4)).to(device)
    X = torch.randn(8, 3, 8, 8)
    y = torch.randint(0, 4, (8,))
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=4)
    optimizer = torch.optim.Adam(model.parameters())
    ckpt = str(tmp_path / "best.pt")
    train_model(model, loader, loader, optimizer, epochs=2, device=device,
                checkpoint_path=ckpt)
    assert os.path.exists(ckpt)
    state = torch.load(ckpt, weights_only=True)
    assert isinstance(state, dict)
