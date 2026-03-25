"""Shared PyTorch utilities: device detection, seeds, training loop."""
import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Return MPS device on Apple Silicon, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)  # torchvision transforms use random.random() internally
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs: int,
    device,
    patience: int = 5,
    checkpoint_path: str = None,
) -> dict:
    """
    Train model and return history dict of per-epoch lists.

    Returns dict with keys: loss, accuracy, val_loss, val_accuracy.
    Each value is a list with one entry per epoch trained.
    """
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # --- Training pass ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)

        # --- Validation pass (inference mode: no grad, no dropout, BN uses running stats) ---
        model.train(mode=False)
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)
        model.train()

        epoch_val_acc = val_correct / val_total
        history["loss"].append(train_loss / train_total)
        history["accuracy"].append(train_correct / train_total)
        history["val_loss"].append(val_loss / val_total)
        history["val_accuracy"].append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}/{epochs}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
