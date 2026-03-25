"""Hyperparameter Analysis Module -- PyTorch implementation."""
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CINIC_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _make_loaders(train_dir, val_dir, batch_size):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    t = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    return (
        DataLoader(datasets.ImageFolder(train_dir, transform=t),
                   batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(datasets.ImageFolder(val_dir, transform=t),
                   batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _get_utils():
    from utils import get_device, train_model
    return get_device(), train_model


def analyze_learning_rates(model_func, train_dir, val_dir,
                            learning_rates=None, epochs=10, batch_size=32):
    if learning_rates is None:
        learning_rates = [0.0001, 0.001, 0.01, 0.1]
    device, train_model = _get_utils()
    train_loader, val_loader = _make_loaders(train_dir, val_dir, batch_size)
    results = []
    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        model = model_func().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = train_model(model, train_loader, val_loader, optimizer,
                              epochs=epochs, device=device)
        results.append({
            "learning_rate": lr,
            "train_accuracy": history["accuracy"][-1],
            "val_accuracy": history["val_accuracy"][-1],
            "train_loss": history["loss"][-1],
            "val_loss": history["val_loss"][-1],
            "epochs": epochs,
        })
    return results


def analyze_batch_sizes(model_func, train_dir, val_dir,
                         batch_sizes=None, epochs=10):
    if batch_sizes is None:
        batch_sizes = [32, 64, 128]
    device, train_model = _get_utils()
    results = []
    for bs in batch_sizes:
        print(f"Testing batch size: {bs}")
        train_loader, val_loader = _make_loaders(train_dir, val_dir, bs)
        model = model_func().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        history = train_model(model, train_loader, val_loader, optimizer,
                              epochs=epochs, device=device)
        results.append({
            "batch_size": bs,
            "train_accuracy": history["accuracy"][-1],
            "val_accuracy": history["val_accuracy"][-1],
            "train_loss": history["loss"][-1],
            "val_loss": history["val_loss"][-1],
            "epochs": epochs,
        })
    return results


def analyze_regularization_strengths(model_func, train_dir, val_dir,
                                       dropout_rates=None, weight_decays=None,
                                       epochs=10, batch_size=32):
    if dropout_rates is None:
        dropout_rates = [0.2, 0.3, 0.5]
    if weight_decays is None:
        weight_decays = [1e-4, 1e-3, 1e-2]
    device, train_model = _get_utils()
    train_loader, val_loader = _make_loaders(train_dir, val_dir, batch_size)
    results = []
    for dr in dropout_rates:
        for wd in weight_decays:
            print(f"Testing dropout={dr}, weight_decay={wd}")
            try:
                # Fresh model + fresh optimizer per iteration.
                # weight_decay flows through the optimizer, not the model.
                model = model_func(dropout_rate=dr, weight_decay=wd).to(device)
                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=0.001, weight_decay=wd)
                history = train_model(model, train_loader, val_loader, optimizer,
                                      epochs=epochs, device=device)
                results.append({
                    "dropout_rate": dr, "weight_decay": wd,
                    "train_accuracy": history["accuracy"][-1],
                    "val_accuracy": history["val_accuracy"][-1],
                    "train_loss": history["loss"][-1],
                    "val_loss": history["val_loss"][-1],
                    "epochs": epochs,
                })
            except Exception as e:
                print(f"Error: {e}")
                results.append({"dropout_rate": dr, "weight_decay": wd,
                                "train_accuracy": 0.0, "val_accuracy": 0.0,
                                "train_loss": 0.0, "val_loss": 0.0, "epochs": epochs})
    return results


def analyze_optimizers(model_func, train_dir, val_dir,
                        optimizers=None, epochs=10, batch_size=32):
    if optimizers is None:
        optimizers = ["adam", "sgd", "rmsprop"]
    device, train_model = _get_utils()
    train_loader, val_loader = _make_loaders(train_dir, val_dir, batch_size)
    results = []
    for opt_name in optimizers:
        print(f"Testing optimizer: {opt_name}")
        model = model_func().to(device)  # fresh model per iteration
        if opt_name == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        elif opt_name == "rmsprop":
            opt = torch.optim.RMSprop(model.parameters(), lr=0.001)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
        try:
            history = train_model(model, train_loader, val_loader, opt,
                                  epochs=epochs, device=device)
            results.append({
                "optimizer": opt_name,
                "train_accuracy": history["accuracy"][-1],
                "val_accuracy": history["val_accuracy"][-1],
                "train_loss": history["loss"][-1],
                "val_loss": history["val_loss"][-1],
                "epochs": epochs,
            })
        except Exception as e:
            print(f"Error: {e}")
            results.append({"optimizer": opt_name, "train_accuracy": 0.0,
                            "val_accuracy": 0.0, "train_loss": 0.0,
                            "val_loss": 0.0, "epochs": epochs})
    return results


def plot_hyperparameter_results(results_df, title="Hyperparameter Analysis Results"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    if "learning_rate" in results_df.columns:
        axes[0, 0].plot(results_df["learning_rate"], results_df["val_accuracy"], "o-")
        axes[0, 0].set_xlabel("Learning Rate"); axes[0, 0].set_xscale("log")
    if "batch_size" in results_df.columns:
        axes[0, 1].bar(results_df["batch_size"].astype(str), results_df["val_accuracy"])
        axes[0, 1].set_xlabel("Batch Size")
    if "dropout_rate" in results_df.columns:
        axes[1, 0].plot(results_df["dropout_rate"], results_df["val_accuracy"], "o-")
        axes[1, 0].set_xlabel("Dropout Rate")
    if "weight_decay" in results_df.columns:
        axes[1, 1].plot(results_df["weight_decay"], results_df["val_accuracy"], "o-")
        axes[1, 1].set_xlabel("Weight Decay"); axes[1, 1].set_xscale("log")
    plt.tight_layout(); plt.show()


def save_hyperparameter_results(results, filename_prefix="hyperparameter_analysis"):
    df = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("results", f"{filename_prefix}_{ts}.csv")
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
    return df


def create_comprehensive_hyperparameter_analysis(model_func, train_dir, val_dir):
    print("Starting comprehensive hyperparameter analysis...")
    lr_results = analyze_learning_rates(
        model_func, train_dir, val_dir, [0.0001, 0.001, 0.01, 0.1], epochs=5)
    batch_results = analyze_batch_sizes(
        model_func, train_dir, val_dir, [16, 32, 64], epochs=5)
    reg_results = analyze_regularization_strengths(
        model_func, train_dir, val_dir,
        [0.1, 0.2, 0.3, 0.5], [1e-4, 1e-3, 1e-2], epochs=5)
    opt_results = analyze_optimizers(
        model_func, train_dir, val_dir, ["adam", "sgd", "rmsprop"], epochs=5)
    print("Hyperparameter analysis complete.")
    return {"learning_rate": lr_results, "batch_size": batch_results,
            "regularization": reg_results, "optimizer": opt_results}
