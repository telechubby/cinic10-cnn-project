# PyTorch Migration Design

**Date:** 2026-03-25
**Scope:** Full TensorFlow → PyTorch migration for GPU acceleration on macOS (Apple Silicon M1 Max)
**Status:** Approved

---

## Background

The project trains CNNs on CINIC-10 using TensorFlow/Keras. TensorFlow does not support GPU on macOS, causing slow CPU-only training and occasional bus errors on Python 3.12+. PyTorch supports Apple Silicon via the MPS (Metal Performance Shaders) backend since PyTorch 1.12, giving full GPU acceleration on M1/M2/M3/M4.

PyTorch and torchvision are already listed in `requirements.txt` but unused. This migration removes all TensorFlow dependencies and replaces them with PyTorch equivalents across all source modules, notebooks, and tests.

---

## Goals

- Enable MPS GPU acceleration on Apple Silicon
- Remove all `tensorflow` and `keras` imports
- Preserve all public function signatures so `main_experiment.py` changes minimally
- Keep module structure (one file per concern)
- Migrate 3 Jupyter notebooks and 3 test files

---

## Architecture

### Section 1: Core Infrastructure — `src/utils.py` (new file)

A new module shared by all other modules.

**Device detection:**
```python
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

**Shared training loop — `train_model()`:**

Replaces Keras `.fit()` across all modules. Signature:
```python
def train_model(
    model, train_loader, val_loader,
    optimizer, epochs, device,
    patience=5, checkpoint_path=None
) -> dict
```

Returns a history dict with keys `loss`, `accuracy`, `val_loss`, `val_accuracy` (same as Keras) so downstream code needs no changes.

Implements:
- Per-epoch training and validation passes
- Early stopping (patience-based, monitors `val_accuracy`)
- Best-weights checkpointing via `torch.save` / `model.load_state_dict`

Loss function: `nn.CrossEntropyLoss` — PyTorch labels are integer class indices (not one-hot), which maps directly to Keras's `categorical_crossentropy`.

---

### Section 2: Model Architecture — `src/model_architecture.py`

All five `create_*` functions return `nn.Module` instances instead of Keras models. Layer mapping:

| Keras | PyTorch |
|---|---|
| `Conv2D(32, (3,3), activation='relu')` | `nn.Conv2d(in_ch, 32, 3, padding=1)` + `nn.ReLU()` |
| `BatchNormalization()` | `nn.BatchNorm2d(channels)` |
| `MaxPooling2D((2,2))` | `nn.MaxPool2d(2)` |
| `Dropout(0.25)` | `nn.Dropout2d(0.25)` |
| `GlobalAveragePooling2D()` | `nn.AdaptiveAvgPool2d(1)` + `nn.Flatten()` |
| `Dense(512, activation='relu')` | `nn.Linear(in, 512)` + `nn.ReLU()` |
| `Dense(10, activation='softmax')` | `nn.Linear(in, 10)` (no softmax — `CrossEntropyLoss` includes log-softmax) |

**L2 regularization** (`create_cnn_with_regularization`): Keras applies `kernel_regularizer=l2(wd)` per layer. In PyTorch, L2 is passed as `weight_decay` to the optimizer — same mathematical effect, cleaner. The public signature `(dropout_rate, weight_decay)` is unchanged; the model stores these for use when building the optimizer.

**Input format**: PyTorch uses channels-first `(B, C, H, W)`. Keras uses `(B, H, W, C)`. The data pipeline handles this transparently via `torchvision.transforms.ToTensor()`.

Public functions unchanged: `create_baseline_cnn`, `create_deep_cnn`, `create_efficient_cnn`, `create_cnn_with_regularization`, `create_few_shot_cnn`.

---

### Section 3: Data Pipeline — `src/data_preprocessing.py`

`ImageDataGenerator.flow_from_directory()` → `torchvision.datasets.ImageFolder` + `torch.utils.data.DataLoader`.

`create_data_generators(train_dir, validation_dir, batch_size, augment)` keeps its exact signature and return value becomes `(train_loader, val_loader)`.

Augmentation transform mapping:

| Keras `ImageDataGenerator` arg | `torchvision.transforms` |
|---|---|
| `rescale=1/255` | `transforms.ToTensor()` |
| `rotation_range=15` | `transforms.RandomRotation(15)` |
| `horizontal_flip=True` | `transforms.RandomHorizontalFlip()` |
| `width_shift_range=0.1, height_shift_range=0.1` | `transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))` |
| `zoom_range=0.1` | `transforms.RandomResizedCrop(32, scale=(0.9, 1.1))` |
| `shear_range=0.1` | `transforms.RandomAffine(degrees=0, shear=10)` |
| `brightness_range=[0.8, 1.2]` | `transforms.ColorJitter(brightness=0.2)` |
| `fill_mode='nearest'` | default behavior of `RandomAffine` |

`normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)` applied after `ToTensor()`.

`subsample_dataset()` uses only `os`, `shutil`, `numpy` — no changes needed.

`load_cinic_data()` uses PIL — no changes needed.

---

### Section 4: Augmentation Studies — `src/augmentation_studies.py`

**`CutoutLayer`** (TF graph op) becomes a plain Python transform class compatible with `torchvision`:
```python
class Cutout:
    def __init__(self, mask_size=8): ...
    def __call__(self, img: torch.Tensor) -> torch.Tensor: ...
```
Applied via `transforms.Compose([..., Cutout(mask_size=8)])`.

**`create_cutout_tf_dataset()`** → **`create_cutout_dataloader()`**: returns a `DataLoader` with the `Cutout` transform in its pipeline instead of a `tf.data.Dataset`.

**`create_standard_augmentation_generators()`** and **`create_advanced_augmentation_generators()`**: return dicts of `transforms.Compose` objects instead of `ImageDataGenerator` instances.

**`evaluate_augmentation_effects()`**: accepts transform configs, creates `DataLoader` per config, uses `train_model()`.

Public API unchanged: `compare_augmentation_approaches(model_func, train_dir, val_dir, epochs, batch_size)`.

---

### Section 5: Hyperparameter Analysis — `src/hyperparameter_analysis.py`

Keras optimizers → `torch.optim`:

| Keras | PyTorch |
|---|---|
| `keras.optimizers.Adam(lr)` | `torch.optim.Adam(model.parameters(), lr=lr)` |
| `keras.optimizers.SGD(lr, momentum=0.9)` | `torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)` |
| `keras.optimizers.RMSprop(lr)` | `torch.optim.RMSprop(model.parameters(), lr=lr)` |

All four `analyze_*` functions keep their signatures. Training uses `train_model()` from `utils.py`.

---

### Section 6: Evaluation — `src/evaluation.py`

**`calculate_performance_metrics(model, test_loader)`**: replaces `model.evaluate()` with a manual eval loop over the `DataLoader`.

**`generate_confusion_matrix(model, test_loader)`**: replaces `model.predict()` with a manual inference loop collecting `torch.argmax` predictions.

**`run_reduced_dataset_experiment()`**: replaces `ImageDataGenerator` with `DataLoader`; uses `train_model()`.

All `plot_*` and `save_*` functions use only matplotlib/pandas — no changes.

---

### Section 7: Few-Shot Learning — `src/few_shot_learning.py`

`create_siamese_network()`, `create_few_shot_classifier()`, `create_prototypical_network()` become `nn.Module` subclasses.

The Siamese distance layer `tf.abs(embedding_1 - embedding_2)` becomes `torch.abs(e1 - e2)`.

`create_few_shot_evaluation()` and `evaluate_few_shot_performance()` keep their signatures; internally use `DataLoader` + `train_model()`.

---

### Section 8: Main Experiment — `src/main_experiment.py`

`keras.callbacks.ModelCheckpoint` and `EarlyStopping` in `run_baseline_experiment()` are removed — the `train_model()` helper handles both. The `model.fit()` call is replaced with `train_model(...)`. All other logic (directory setup, CSV saving, summary writing) is unchanged.

---

### Section 9: Tests

Three test files updated to use PyTorch types:
- Construct `nn.Module` instances via the `create_*` functions
- Pass `torch.Tensor` inputs with shape `(B, C, H, W)`
- Assert output tensor shapes and dtypes
- Test `DataLoader` creation instead of generator creation
- Same test intent, PyTorch types

---

### Section 10: Notebooks

Three notebooks (`baseline_experiment.ipynb`, `hyperparameter_tuning.ipynb`, `augmentation_analysis.ipynb`):
- Import cells updated: remove `tensorflow`/`keras` imports, add `torch`/`torchvision`
- Any inline Keras/TF code in cells updated to PyTorch equivalents
- Public function calls to `src/` modules are unchanged (API preserved)
- No structural changes to notebook flow

---

## Data Flow

```
data/train/ (ImageFolder)
    → torchvision.transforms.Compose (resize, augment, ToTensor, Normalize)
    → DataLoader (batch_size=32, shuffle=True)
    → model.forward(batch.to(device))   # device = mps
    → nn.CrossEntropyLoss
    → optimizer.step()
    → train_model() returns history dict
```

---

## Key Differences from Keras

| Concern | Keras | PyTorch |
|---|---|---|
| GPU device | automatic | explicit `.to(device)` |
| Label format | one-hot | integer class index |
| Loss | `categorical_crossentropy` | `nn.CrossEntropyLoss` |
| Training | `model.fit()` | manual loop in `train_model()` |
| L2 regularization | per-layer `kernel_regularizer` | `weight_decay` in optimizer |
| Data format | HWC `(B,H,W,C)` | CHW `(B,C,H,W)` |
| Softmax | output layer activation | inside `CrossEntropyLoss` |
| Checkpointing | `ModelCheckpoint` callback | `torch.save(state_dict)` |
| Early stopping | `EarlyStopping` callback | logic in `train_model()` |

---

## Dependencies

Remove from `requirements.txt`:
```
tensorflow>=2.21.0
```

Keep/add:
```
torch>=2.4.0
torchvision>=0.19.0
```

No other dependency changes needed.

---

## File Change Summary

| File | Change |
|---|---|
| `src/utils.py` | New — device detection + `train_model()` |
| `src/model_architecture.py` | Full rewrite — `nn.Module` |
| `src/data_preprocessing.py` | Partial — `create_data_generators()` only |
| `src/augmentation_studies.py` | Full rewrite — transforms + DataLoader |
| `src/hyperparameter_analysis.py` | Full rewrite — torch.optim + train_model() |
| `src/evaluation.py` | Partial — eval/predict loops + DataLoader |
| `src/few_shot_learning.py` | Full rewrite — nn.Module + train_model() |
| `src/main_experiment.py` | Minimal — remove callbacks, use train_model() |
| `tests/test_data_preprocessing.py` | Update to PyTorch types |
| `tests/test_evaluation.py` | Update to PyTorch types |
| `tests/test_augmentation_studies.py` | Update to PyTorch types |
| `notebooks/baseline_experiment.ipynb` | Update imports + inline TF code |
| `notebooks/hyperparameter_tuning.ipynb` | Update imports + inline TF code |
| `notebooks/augmentation_analysis.ipynb` | Update imports + inline TF code |
