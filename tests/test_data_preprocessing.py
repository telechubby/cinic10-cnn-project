"""Tests for data_preprocessing utility functions."""
import os
import shutil
import tempfile
import sys

# Set up path before importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pytest
from PIL import Image

from data_preprocessing import subsample_dataset, CINIC_CLASSES


def make_mock_dataset(root_dir, classes, n_per_class=20):
    """Create a mock dataset directory structure with dummy PNG images."""
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            img.save(os.path.join(cls_dir, f"img_{i:04d}.png"))


def test_subsample_dataset_fraction():
    """subsample_dataset with fraction=0.5 copies ~50% of images per class."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "source")
        dst = os.path.join(tmpdir, "dest")
        classes = ["cat", "dog"]
        n_per_class = 20
        make_mock_dataset(src, classes, n_per_class)

        subsample_dataset(src, dst, fraction=0.5)

        for cls in classes:
            copied = len(os.listdir(os.path.join(dst, cls)))
            assert copied == 10, f"Expected 10 images for {cls}, got {copied}"


def test_subsample_dataset_n_per_class():
    """subsample_dataset with n_per_class=5 copies exactly 5 images per class."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "source")
        dst = os.path.join(tmpdir, "dest")
        classes = ["airplane", "ship"]
        make_mock_dataset(src, classes, n_per_class=20)

        subsample_dataset(src, dst, n_per_class=5)

        for cls in classes:
            copied = len(os.listdir(os.path.join(dst, cls)))
            assert copied == 5, f"Expected 5 images for {cls}, got {copied}"


def test_subsample_dataset_creates_class_subdirs():
    """subsample_dataset creates class subdirectories in dest even if dest doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "source")
        dst = os.path.join(tmpdir, "nonexistent_dest")
        classes = ["frog"]
        make_mock_dataset(src, classes, n_per_class=10)

        subsample_dataset(src, dst, fraction=1.0)

        assert os.path.isdir(os.path.join(dst, "frog"))


def test_subsample_dataset_requires_fraction_or_n():
    """subsample_dataset raises ValueError if neither fraction nor n_per_class is given."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError):
            subsample_dataset(tmpdir, tmpdir)


def test_create_data_generators_returns_dataloaders():
    import os, tempfile
    import numpy as np
    from PIL import Image
    import torch
    from torch.utils.data import DataLoader
    from data_preprocessing import create_data_generators

    with tempfile.TemporaryDirectory() as tmpdir:
        for split in ["train", "val"]:
            for cls in ["cat", "dog"]:
                cls_dir = os.path.join(tmpdir, split, cls)
                os.makedirs(cls_dir)
                for i in range(5):
                    img = Image.fromarray(
                        np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
                    img.save(os.path.join(cls_dir, f"img_{i}.png"))

        train_loader, val_loader = create_data_generators(
            os.path.join(tmpdir, "train"),
            os.path.join(tmpdir, "val"),
            batch_size=4, augment=False,
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        images, labels = next(iter(train_loader))
        assert images.shape[1] == 3   # channels first (C,H,W)
        assert images.shape[2] == 32
        assert images.shape[3] == 32
        assert labels.dtype == torch.long


def test_create_data_generators_augment_flag():
    import os, tempfile
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader
    from data_preprocessing import create_data_generators

    with tempfile.TemporaryDirectory() as tmpdir:
        for split in ["train", "val"]:
            for cls in ["cat", "dog"]:
                cls_dir = os.path.join(tmpdir, split, cls)
                os.makedirs(cls_dir)
                for i in range(5):
                    img = Image.fromarray(
                        np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
                    img.save(os.path.join(cls_dir, f"img_{i}.png"))

        for augment in [True, False]:
            train_loader, val_loader = create_data_generators(
                os.path.join(tmpdir, "train"),
                os.path.join(tmpdir, "val"),
                batch_size=4, augment=augment,
            )
            assert isinstance(train_loader, DataLoader)
