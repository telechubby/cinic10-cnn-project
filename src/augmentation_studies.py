"""Data Augmentation Studies Module -- PyTorch/torchvision implementation."""
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CINIC_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class Cutout:
    """
    Torchvision-compatible transform: zeros a random square patch.
    Input: (C, H, W) float tensor from ToTensor().
    Mask: img[:, y1:y2, x1:x2] = 0  (dims 1=height, 2=width).
    """
    def __init__(self, mask_size: int = 8):
        self.mask_size = mask_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        half = self.mask_size // 2
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        y1, y2 = max(0, cy - half), min(h, cy + half)
        result = img.clone()
        result[:, y1:y2, x1:x2] = 0.0
        return result


def _val_transform():
    return transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])


def create_standard_augmentation_generators() -> dict:
    """Return dict of torchvision.transforms.Compose for standard augmentations."""
    return {
        "standard": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1), shear=5.7),  # 0.1 rad ≈ 5.7°
            transforms.ToTensor(),
        ]),
        "color_jitter": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15),
                                    scale=(0.8, 1.2), shear=11.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]),
        "minimal": transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor()]),
        "crop_resize": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                    scale=(0.95, 1.05), shear=2.9),
            transforms.ToTensor(),
        ]),
    }


def create_advanced_augmentation_generators() -> dict:
    """Return dict of augmentation configs (Compose or placeholder None for cutout)."""
    return {
        "cutout": None,  # filled by compare_augmentation_approaches
        "mixup": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ]),
        "autoaugment_like": transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15),
                                    scale=(0.85, 1.15), shear=8.6),
            transforms.ToTensor(),
        ]),
    }


def create_cutout_dataloader(train_dir: str, batch_size: int = 32,
                             mask_size: int = 8) -> DataLoader:
    """Return a DataLoader with Cutout augmentation (replaces create_cutout_tf_dataset)."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(mask_size=mask_size),
    ])
    return DataLoader(datasets.ImageFolder(train_dir, transform=transform),
                      batch_size=batch_size, shuffle=True, num_workers=0)


def apply_cutout_augmentation(image: np.ndarray, mask_size: int = 8) -> np.ndarray:
    augmented = np.copy(image)
    h, w, _ = image.shape
    y, x = np.random.randint(0, h), np.random.randint(0, w)
    y1, y2 = max(0, y - mask_size // 2), min(h, y + mask_size // 2)
    x1, x2 = max(0, x - mask_size // 2), min(w, x + mask_size // 2)
    augmented[y1:y2, x1:x2] = 0
    return augmented


def apply_cutmix_augmentation(image1, image2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    h, w, _ = image1.shape
    crop_h, crop_w = int(h * lam), int(w * lam)
    y_start = np.random.randint(0, max(1, h - crop_h))
    x_start = np.random.randint(0, max(1, w - crop_w))
    mixed = np.copy(image1)
    mixed[y_start:y_start + crop_h, x_start:x_start + crop_w] = \
        image2[y_start:y_start + crop_h, x_start:x_start + crop_w]
    return mixed


def evaluate_augmentation_effects(model_func, train_dir, val_dir,
                                   augmentation_configs, epochs=5, batch_size=32):
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from utils import get_device, train_model

    device = get_device()
    val_loader = DataLoader(datasets.ImageFolder(val_dir, transform=_val_transform()),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    results = []
    for aug_name, aug_config in augmentation_configs.items():
        if aug_config is None:
            continue
        print(f"Evaluating {aug_name} augmentation...")
        try:
            if isinstance(aug_config, DataLoader):
                train_loader = aug_config
            else:
                train_loader = DataLoader(
                    datasets.ImageFolder(train_dir, transform=aug_config),
                    batch_size=batch_size, shuffle=True, num_workers=0)
            model = model_func().to(device)
            optimizer = torch.optim.Adam(model.parameters())
            history = train_model(model, train_loader, val_loader, optimizer,
                                  epochs=epochs, device=device)
            results.append({
                "augmentation": aug_name,
                "train_accuracy": history["accuracy"][-1],
                "val_accuracy": history["val_accuracy"][-1],
                "train_loss": history["loss"][-1],
                "val_loss": history["val_loss"][-1],
                "epochs": epochs,
            })
        except Exception as e:
            print(f"Error with {aug_name}: {e}")
            results.append({"augmentation": aug_name, "train_accuracy": 0.0,
                            "val_accuracy": 0.0, "train_loss": 0.0,
                            "val_loss": 0.0, "epochs": epochs})
    return results


def visualize_augmentation_results(results_df, title="Data Augmentation Analysis"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    for ax, col, color in [
        (axes[0, 0], "val_accuracy", "skyblue"),
        (axes[0, 1], "train_accuracy", "lightgreen"),
        (axes[1, 0], "val_loss", "lightcoral"),
        (axes[1, 1], "train_loss", "lightyellow"),
    ]:
        ax.bar(results_df["augmentation"], results_df[col], color=color)
        ax.set_title(f"Augmentation vs {col}")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


def save_augmentation_results(results, filename_prefix="augmentation_analysis"):
    df = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("results", f"{filename_prefix}_{ts}.csv")
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
    return df


def compare_augmentation_approaches(model_func, train_dir, val_dir,
                                     epochs=5, batch_size=32):
    print("Starting comprehensive augmentation analysis...")
    standard_results = evaluate_augmentation_effects(
        model_func, train_dir, val_dir,
        create_standard_augmentation_generators(), epochs=epochs, batch_size=batch_size)
    advanced = create_advanced_augmentation_generators()
    advanced["cutout"] = create_cutout_dataloader(train_dir, batch_size=batch_size)
    advanced_results = evaluate_augmentation_effects(
        model_func, train_dir, val_dir, advanced, epochs=epochs, batch_size=batch_size)
    print("Augmentation analysis complete.")
    return {"standard": standard_results, "advanced": advanced_results}
