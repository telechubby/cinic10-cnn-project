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

Returns a history dict where each key maps to a **list of per-epoch values** matching the Keras `history.history` interface (e.g. `{"loss": [0.9, 0.7, ...], "val_accuracy": [0.4, 0.6, ...]}`). Downstream code that accesses `history["val_accuracy"][-1]` continues to work unchanged.

Implements:
- Per-epoch training pass: `model.train()` then forward/loss/backward/step
- Per-epoch validation pass: `model.eval()` + `torch.no_grad()` block to accumulate loss and accuracy, then `model.train()` to resume
- Early stopping (patience-based, monitors `val_accuracy`)
- Best-weights checkpointing via `torch.save(model.state_dict())` / `model.load_state_dict()`

Loss function: `nn.CrossEntropyLoss` — PyTorch labels are integer class indices (not one-hot), which maps directly to Keras's `categorical_crossentropy`.

**Random seeds**: `utils.py` exports a `set_seeds(seed=42)` helper that calls `torch.manual_seed(seed)`, `torch.mps.manual_seed(seed)`, and `np.random.seed(seed)`. Called once at the top of `main_experiment.py`.

**MPS fallback**: Set env var `PYTORCH_ENABLE_MPS_FALLBACK=1` so any MPS-unsupported ops silently fall back to CPU instead of raising errors.

---

### Section 2: Model Architecture — `src/model_architecture.py`

All five `create_*` functions return `nn.Module` instances instead of Keras models. Layer mapping:

| Keras | PyTorch |
|---|---|
| `Conv2D(32, (3,3), activation='relu')` | `nn.Conv2d(in_ch, 32, 3, padding=1)` + `nn.ReLU()` |
| `BatchNormalization()` | `nn.BatchNorm2d(channels)` |
| `MaxPooling2D((2,2))` | `nn.MaxPool2d(2)` |
| `Dropout(rate)` after conv layer | `nn.Dropout2d(rate)` |
| `Dropout(rate)` after `Dense`/`Linear` | `nn.Dropout(rate)` |
| `GlobalAveragePooling2D()` | `nn.AdaptiveAvgPool2d(1)` + `nn.Flatten()` |
| `Dense(512, activation='relu')` | `nn.Linear(in, 512)` + `nn.ReLU()` |
| `Dense(10, activation='softmax')` | `nn.Linear(in, 10)` (no softmax — `CrossEntropyLoss` includes log-softmax) |

**L2 regularization** (`create_cnn_with_regularization`): Keras applies `kernel_regularizer=l2(wd)` per layer. In PyTorch, L2 is passed as `weight_decay` to the optimizer — same mathematical effect, cleaner. The public signature `(dropout_rate, weight_decay)` is unchanged. The function returns an `(model, weight_decay)` tuple, or callers (e.g. `hyperparameter_analysis.py`) construct the optimizer themselves: `torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)`. The model does **not** store `weight_decay` internally.

**`get_model_summary(model)`**: `model.summary()` is Keras-specific. Replace with a `try/except ImportError` block: attempt `torchinfo.summary(model, input_size=(1, 3, 32, 32))`, fall back to `print(model)` if `torchinfo` is not installed. This never raises for a missing optional dependency. Add `torchinfo>=1.8.0` as an optional dev dependency.

**Input format**: PyTorch uses channels-first `(B, C, H, W)`. Keras uses `(B, H, W, C)`. The data pipeline handles this transparently via `torchvision.transforms.ToTensor()`.

Public functions unchanged: `create_baseline_cnn`, `create_deep_cnn`, `create_efficient_cnn`, `create_cnn_with_regularization`, `create_few_shot_cnn`.

---

### Section 3: Data Pipeline — `src/data_preprocessing.py`

`ImageDataGenerator.flow_from_directory()` → `torchvision.datasets.ImageFolder` + `torch.utils.data.DataLoader`.

`create_data_generators(train_dir, validation_dir, batch_size, augment)` keeps its exact signature and return value becomes `(train_loader, val_loader)`.

Augmentation transform mapping:

| Keras `ImageDataGenerator` arg | `torchvision.transforms` |
|---|---|
| `rescale=1/255` | `transforms.ToTensor()` (divides by 255, outputs float32 in [0,1]) |
| `rotation_range=15` | `transforms.RandomRotation(15)` |
| `horizontal_flip=True` | `transforms.RandomHorizontalFlip()` |
| `width_shift_range=0.1, height_shift_range=0.1` | `transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))` |
| `zoom_range=0.1` | `transforms.RandomAffine(degrees=0, scale=(0.9, 1.1))` |
| `shear_range=0.1` (radians) | `transforms.RandomAffine(degrees=0, shear=5.7)` (degrees: 0.1 rad ≈ 5.7°) |
| `brightness_range=[0.8, 1.2]` | `transforms.ColorJitter(brightness=0.2)` |
| `fill_mode='nearest'` | default behavior of `RandomAffine` |

**No `Normalize` call** — to match the existing TF baseline which uses only `rescale=1/255` with no channel-wise normalization, `ToTensor()` alone is used. This keeps behaviour identical to the current pipeline. If channel-wise normalization is added later, use CINIC-10's published stats: `mean=[0.47889522, 0.47227842, 0.43047404]`, `std=[0.24205776, 0.23828046, 0.25874835]`.

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

`__call__` receives a `(C, H, W)` tensor (output of `ToTensor()`). The mask zeros channels-last via `img[:, y1:y2, x1:x2] = 0` — dims 1 and 2 are height/width, not dim 0.

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

**Per-iteration model + optimizer re-creation**: each loop iteration creates a fresh model and a fresh optimizer bound to that model's parameters before calling `train_model()`. A single optimizer must never be shared across iterations.

**`analyze_regularization_strengths`**: receives `dropout_rate` and `weight_decay` from the loop, creates a model via `model_func(dropout_rate=dr, weight_decay=wd)`, then constructs `torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=wd)` before calling `train_model()`. The `weight_decay` value flows through the optimizer, not through the model.

**No LR scheduler**: `train_model()` does not manage a learning rate scheduler. Callers who need one step it externally. This keeps the signature stable.

---

### Section 6: Evaluation — `src/evaluation.py`

**`calculate_performance_metrics(model, test_loader)`**: replaces `model.evaluate()` with a manual eval loop. Must call `model.eval()` before the loop and wrap iterations in `torch.no_grad()` to prevent BatchNorm stat updates, Dropout activation, and unnecessary gradient graph construction on MPS.

**`generate_confusion_matrix(model, test_loader)`**: same requirements — `model.eval()` + `torch.no_grad()`. Each `DataLoader` batch yields `(images, labels)` tuples; true labels are accumulated from the second element (`labels`, a `LongTensor` of integer class indices) and predictions from `torch.argmax(output, dim=1)`. Final confusion matrix computed via `sklearn.metrics.confusion_matrix(true_labels, predicted_labels)`.

**`compare_model_performance(models_dict, test_loader)`**: currently stubbed with simulated values. After migration, the argument type changes from `test_generator` to `test_loader` (a `DataLoader`). The simulated logic can remain for now but the signature is updated.

**`run_reduced_dataset_experiment()`**: replaces `ImageDataGenerator` with `DataLoader`; uses `train_model()`.

All `plot_*` and `save_*` functions use only matplotlib/pandas — no changes.

---

### Section 7: Few-Shot Learning — `src/few_shot_learning.py`

`create_siamese_network()`, `create_few_shot_classifier()`, `create_prototypical_network()` become `nn.Module` subclasses.

Keras's multi-input functional API has no direct `nn.Module` equivalent. `create_siamese_network()` returns an `nn.Module` subclass with a `forward(self, x1, x2)` method: both inputs pass through the shared backbone, then `dist = torch.abs(e1 - e2)` feeds into `nn.Linear(embedding_dim, 1)` + `nn.Sigmoid()`. The shared backbone is stored as `self.backbone` so weights are tied.

`create_few_shot_evaluation()` and `evaluate_few_shot_performance()` keep their signatures; internally use `DataLoader` + `train_model()`.

---

### Section 8: Main Experiment — `src/main_experiment.py`

`keras.callbacks.ModelCheckpoint` and `EarlyStopping` in `run_baseline_experiment()` are removed — the `train_model()` helper handles both. The `model.fit()` call is replaced with `train_model(...)`. All other logic (directory setup, CSV saving, summary writing) is unchanged.

---

### Section 9: Tests

Three existing test files updated to use PyTorch types:
- Construct `nn.Module` instances via the `create_*` functions
- Pass `torch.Tensor` inputs with shape `(B, C, H, W)`
- Assert output tensor shapes and dtypes
- Test `DataLoader` creation instead of generator creation
- Same test intent, PyTorch types

Two new test files added for currently untested modules:
- `tests/test_model_architecture.py`: for each `create_*` function, feed `torch.zeros(1, 3, 32, 32)` and assert `output.shape == (1, 10)`. Verifies channels-first layout and correct `Dropout` vs `Dropout2d` usage.
- `tests/test_few_shot_learning.py`: forward-pass test for `create_siamese_network` with two inputs (`x1, x2`), and for `create_prototypical_network` / `create_few_shot_classifier`.

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
    → torchvision.transforms.Compose (resize, augment, ToTensor)  # no Normalize
    → DataLoader (batch_size=32, shuffle=True)
    → model.train() → model.forward(batch.to(device))   # device = mps
    → nn.CrossEntropyLoss
    → optimizer.step()
    [validation] → model.eval() + torch.no_grad() → accumulate metrics → model.train()
    → train_model() returns history dict {"loss": [...], "val_accuracy": [...], ...}
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
torchinfo>=1.8.0   # optional, for get_model_summary()
```

Also update the file change summary table to add the two new test files.

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
| `tests/test_model_architecture.py` | New — forward-pass shape tests for all 5 architectures |
| `tests/test_few_shot_learning.py` | New — forward-pass tests incl. Siamese two-input |
| `notebooks/baseline_experiment.ipynb` | Update imports + inline TF code |
| `notebooks/hyperparameter_tuning.ipynb` | Update imports + inline TF code |
| `notebooks/augmentation_analysis.ipynb` | Update imports + inline TF code |
