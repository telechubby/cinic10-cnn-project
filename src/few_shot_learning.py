"""Few-Shot Learning Module -- PyTorch nn.Module implementations."""
import os
import sys
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CINIC_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _backbone(embedding_dim=128):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, embedding_dim), nn.ReLU(),
    )


class SiameseNetwork(nn.Module):
    """
    Siamese network with shared backbone.
    forward(x1, x2) -> (B, 1) sigmoid similarity score.
    self.backbone is the shared weight module.
    """
    def __init__(self, input_shape=(32, 32, 3), embedding_dim=128):
        super().__init__()
        self.backbone = _backbone(embedding_dim)
        self.similarity = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())

    def forward(self, x1, x2):
        e1, e2 = self.backbone(x1), self.backbone(x2)
        return self.similarity(torch.abs(e1 - e2))


def create_siamese_network(input_shape=(32, 32, 3), embedding_dim=128):
    return SiameseNetwork(input_shape=input_shape, embedding_dim=embedding_dim)


class FewShotClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_few_shot_classifier(input_shape=(32, 32, 3), num_classes=10):
    return FewShotClassifier(num_classes=num_classes)


class PrototypicalNetwork(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=128):
        super().__init__()
        self._backbone = _backbone(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.classifier(self._backbone(x))


def create_prototypical_network(input_shape=(32, 32, 3), num_classes=10,
                                 embedding_dim=128):
    return PrototypicalNetwork(num_classes=num_classes, embedding_dim=embedding_dim)


def create_few_shot_evaluation(model_func, train_dir, val_dir,
                                few_shot_samples=None, epochs=10, batch_size=32):
    from data_preprocessing import subsample_dataset
    from utils import get_device, train_model
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    if few_shot_samples is None:
        few_shot_samples = [1, 5, 10]

    device = get_device()
    val_t = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    val_loader = DataLoader(datasets.ImageFolder(val_dir, transform=val_t),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    results = []
    for n in few_shot_samples:
        print(f"Evaluating few-shot with {n} samples per class...")
        with tempfile.TemporaryDirectory() as tmp_train:
            subsample_dataset(train_dir, tmp_train, n_per_class=n)
            actual_batch = min(batch_size, n * 10)
            train_t = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            train_loader = DataLoader(
                datasets.ImageFolder(tmp_train, transform=train_t),
                batch_size=actual_batch, shuffle=True, num_workers=0)
            model = model_func().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            history = train_model(model, train_loader, val_loader, optimizer,
                                  epochs=epochs, device=device)
            val_acc = history["val_accuracy"][-1] if history["val_accuracy"] else 0.0
            val_loss = history["val_loss"][-1] if history["val_loss"] else 0.0
            results.append({
                "samples_per_class": n,
                "train_accuracy": history["accuracy"][-1] if history["accuracy"] else 0.0,
                "val_accuracy": val_acc,
                "train_loss": history["loss"][-1] if history["loss"] else 0.0,
                "val_loss": val_loss,
                "epochs": epochs,
            })
            print(f"  n={n}: val_acc={results[-1]['val_accuracy']:.4f}")
    return results


def evaluate_few_shot_performance(model_func, train_dir, val_dir,
                                   few_shot_configs=None, epochs=10, batch_size=32):
    print("Starting few-shot learning performance evaluation...")
    if few_shot_configs is None:
        few_shot_configs = [1, 5, 10]
    results = create_few_shot_evaluation(
        model_func, train_dir=train_dir, val_dir=val_dir,
        few_shot_samples=few_shot_configs, epochs=epochs, batch_size=batch_size)
    print("Few-shot evaluation complete.")
    return {"few_shot": results}


def plot_few_shot_results(results_df, title="Few-Shot Learning Performance"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    for ax, col, color in [
        (axes[0, 0], "val_accuracy", "blue"),
        (axes[0, 1], "train_accuracy", "green"),
        (axes[1, 0], "val_loss", "red"),
        (axes[1, 1], "train_loss", "orange"),
    ]:
        if "samples_per_class" in results_df.columns:
            ax.plot(results_df["samples_per_class"], results_df[col], "o-", color=color)
            ax.set_xlabel("Samples per Class"); ax.set_ylabel(col); ax.grid(True)
    plt.tight_layout(); plt.show()


def save_few_shot_results(results, filename_prefix="few_shot_analysis"):
    flat = []
    for r in results.get("few_shot", []):
        flat.append({"model_type": "Few-Shot Classifier",
                     "samples_per_class": r.get("samples_per_class", 0),
                     **{k: r.get(k, 0.0) for k in
                        ["train_accuracy", "val_accuracy", "train_loss",
                         "val_loss", "epochs"]}})
    df = pd.DataFrame(flat)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("results", f"{filename_prefix}_{ts}.csv")
    df.to_csv(path, index=False)
    print(f"Few-shot results saved to {path}")
    return df
