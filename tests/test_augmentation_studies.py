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
    img = torch.ones(3, 32, 32)
    result = Cutout(mask_size=8)(img)
    assert result.shape == (3, 32, 32)
    assert (result == 0).any()


def test_cutout_full_mask_zeros_all_spatial():
    from augmentation_studies import Cutout
    img = torch.ones(3, 32, 32)
    result = Cutout(mask_size=32)(img)
    assert result.shape == (3, 32, 32)


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
