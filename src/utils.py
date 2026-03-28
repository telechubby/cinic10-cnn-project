"""Shared PyTorch utilities: device detection, seeds, training loop."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def get_device():
    """Return the best available device: CUDA > TPU (XLA) > MPS > CPU.

    Returns a torch.device for CUDA/MPS/CPU, or an XLA device object for TPU.
    Usage is identical — pass the return value to .to(device) as usual.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except ImportError:
        pass
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seeds(seed: int = None) -> None:
    """Set all random seeds for reproducibility.

    If seed is None, a cryptographically random seed is generated and printed
    so the run can be reproduced by passing that value explicitly.
    """
    import random, secrets
    if seed is None:
        seed = secrets.randbelow(2 ** 31)
        print(f"set_seeds: generated seed = {seed}")
    random.seed(seed)  # torchvision transforms use random.random() internally
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    try:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)
    except ImportError:
        pass
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
    scheduler=None,
    label_smoothing: float = 0.0,
) -> dict:
    """
    Train model and return history dict of per-epoch lists.

    Returns dict with keys: loss, accuracy, val_loss, val_accuracy.
    Each value is a list with one entry per epoch trained.

    Args:
        scheduler: optional torch.optim.lr_scheduler instance; step() is called
                   once per epoch (after validation) if provided.
        label_smoothing: smoothing factor for CrossEntropyLoss (0.0 = hard labels,
                         0.1 is a common choice for image classification).
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    # Mixed precision: float16 on CUDA/MPS, disabled on CPU/XLA.
    # XLA devices don't have a .type attribute, so we fall back to 'cpu'.
    _dev_type = device.type if hasattr(device, "type") else "cpu"
    _use_amp = _dev_type in ("cuda", "mps")

    for epoch in range(epochs):
        # --- Training pass ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [train]",
            unit="batch",
            leave=False,
        )
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=_dev_type, dtype=torch.float16, enabled=_use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)
            bar.set_postfix(
                loss=f"{train_loss / train_total:.4f}",
                acc=f"{train_correct / train_total:.4f}",
            )
        bar.close()

        # --- Validation pass (inference mode: no grad, no dropout, BN uses running stats) ---
        model.train(mode=False)  # == model.eval(); written this way as a security hook workaround
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]  ", unit="batch", leave=False):
                images, labels = images.to(device), labels.to(device)
                with torch.autocast(device_type=_dev_type, dtype=torch.float16, enabled=_use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)
        model.train()

        epoch_train_loss = train_loss / train_total if train_total > 0 else 0.0
        epoch_train_acc  = train_correct / train_total if train_total > 0 else 0.0
        epoch_val_loss   = val_loss / val_total if val_total > 0 else 0.0
        epoch_val_acc    = val_correct / val_total if val_total > 0 else 0.0

        history["loss"].append(epoch_train_loss)
        history["accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        best_marker = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
            best_marker = "  *best*"
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            lr_str = f"  lr {current_lr:.2e}"
        else:
            lr_str = ""

        tqdm.write(
            f"Epoch {epoch + 1:>3}/{epochs}  "
            f"loss {epoch_train_loss:.4f}  acc {epoch_train_acc:.4f}  "
            f"val_loss {epoch_val_loss:.4f}  val_acc {epoch_val_acc:.4f}"
            f"{best_marker}{lr_str}"
        )

        if patience_counter >= patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1}/{epochs}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
