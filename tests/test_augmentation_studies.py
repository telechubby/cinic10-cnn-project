"""Tests for augmentation_studies module."""
import os, sys, inspect
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_cutout_class_exists():
    from augmentation_studies import Cutout
    assert Cutout is not None


def test_cutout_mask_size_attribute():
    from augmentation_studies import Cutout
    assert Cutout(mask_size=16).mask_size == 16


def test_cutout_zeros_patch_on_chw_tensor():
    """Cutout sets a square region to 0 on a (C,H,W) tensor via img[:,y1:y2,x1:x2]=0."""
    from augmentation_studies import Cutout
    np.random.seed(42)
    img = torch.ones(3, 32, 32)
    cutout = Cutout(mask_size=8)
    result = cutout(img)
    assert result.shape == (3, 32, 32)
    assert (result == 0).any(), "Expected some pixels to be zeroed"
    # Verify zeroed region spans all 3 channels (not just one channel)
    # Find any zeroed pixel and check all channels are zero there
    zero_mask = (result == 0)
    if zero_mask.any():
        # Find a spatial location that is zero
        zero_positions = zero_mask[0].nonzero(as_tuple=False)  # (N, 2) on first channel
        if len(zero_positions) > 0:
            h_idx, w_idx = zero_positions[0]
            assert (result[:, h_idx, w_idx] == 0).all(), \
                "Zeroed spatial location should be zero across ALL channels"
    # Verify original image is unmodified (clone behavior)
    assert (img == 1).all(), "Original image must not be modified (Cutout should clone)"


def test_cutout_full_mask_zeros_all_spatial():
    from augmentation_studies import Cutout
    img = torch.ones(3, 32, 32)
    result = Cutout(mask_size=32)(img)
    assert result.shape == (3, 32, 32)
    # With mask_size=32, the entire spatial region should be zeroed
    assert (result == 0).all(), "Expected full-size mask to zero all spatial pixels"


def test_evaluate_augmentation_effects_signature():
    from augmentation_studies import evaluate_augmentation_effects
    sig = inspect.signature(evaluate_augmentation_effects)
    assert "train_dir" in sig.parameters
    assert "val_dir" in sig.parameters


def test_compare_augmentation_approaches_signature():
    from augmentation_studies import compare_augmentation_approaches
    sig = inspect.signature(compare_augmentation_approaches)
    assert "train_dir" in sig.parameters
    assert "val_dir" in sig.parameters


def test_standard_generators_returns_dict_of_transforms():
    from torchvision import transforms
    from augmentation_studies import create_standard_augmentation_generators
    configs = create_standard_augmentation_generators()
    assert isinstance(configs, dict)
    assert "standard" in configs
    for v in configs.values():
        assert isinstance(v, transforms.Compose)


def test_no_validation_split_in_generators():
    from augmentation_studies import create_standard_augmentation_generators
    source = inspect.getsource(create_standard_augmentation_generators)
    assert "validation_split" not in source


def test_create_cutout_dataloader_returns_dataloader():
    import os, tempfile
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader
    from augmentation_studies import create_cutout_dataloader

    with tempfile.TemporaryDirectory() as tmpdir:
        for cls in ["cat", "dog"]:
            cls_dir = os.path.join(tmpdir, cls)
            os.makedirs(cls_dir)
            for i in range(3):
                img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
                img.save(os.path.join(cls_dir, f"img_{i}.png"))
        loader = create_cutout_dataloader(tmpdir, batch_size=4, mask_size=8)
        assert isinstance(loader, DataLoader)
        images, labels = next(iter(loader))
        assert images.shape[1] == 3  # CHW channels-first
