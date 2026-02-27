# VisionT

Vision Transformer training and evaluation pipeline for the NEU surface defect dataset (`NEU-CLS-64`), built with PyTorch and managed with `uv`.

## What this repo does

- Trains a ViT-style classifier on grayscale steel defect images.
- Uses data augmentation (RandAugment, flips, rotation), mixup/cutmix, AdamW, and cosine warmup scheduling.
- Saves checkpoints, TensorBoard logs, and evaluation plots/CSVs.

## Project layout

- `train.py`: training loop, checkpointing, TensorBoard logging, training curves.
- `evaluate.py`: loads `checkpoints/best.pth` and generates test metrics + plots.
- `dataset.py`: dataset splitting and transforms.
- `model.py`: model definition (`SteelViT`).
- `config.py`: all configurable settings (paths, hyperparameters, device, epochs, etc.).
- `pyproject.toml`: dependencies and `uv` settings (includes PyTorch wheel index).

## Prerequisites

- Python `3.11` to `3.13` (as defined in `pyproject.toml`).
- `uv` installed: https://docs.astral.sh/uv/
- Dataset available at `./NEU-CLS-64`.

Dataset must follow `torchvision.datasets.ImageFolder` format:

```text
NEU-CLS-64/
  class_1/
    img1.jpg
    img2.jpg
  class_2/
    ...
```

## Quick start with uv

From repo root:

```powershell
# 1) Create/update virtual environment and install all dependencies from lockfile
uv sync

# 2) Run training
uv run python train.py

# 3) Run evaluation (expects checkpoints/best.pth from training)
uv run python evaluate.py
```

## VS Code workflow

- Open repo in VS Code (`code .`).
- Select interpreter: `Ctrl+Shift+P` -> `Python: Select Interpreter` -> choose `.venv`.
- Run scripts from terminal with `uv run ...` (recommended) so dependencies/interpreter stay consistent.

## Useful uv commands

```powershell
# Install dependencies from pyproject + uv.lock
uv sync

# Install with dev extras if added in future
uv sync --dev

# Add a new dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Run a command inside project environment
uv run <command>

# Show dependency tree
uv tree

# Re-lock dependency versions
uv lock
```

## Configuration

Main knobs are in `config.py`:

- Paths: `DATA_DIR`, `CHECKPOINT_DIR`, `LOG_DIR`
- Training: `BATCH_SIZE`, `NUM_EPOCHS`, `LR`, `WEIGHT_DECAY`
- Device: `DEVICE` (`"cuda"` or `"cpu"`), `AMP`
- Data split: `TRAIN_SPLIT`, `VAL_SPLIT`, `TEST_SPLIT`

If you run on CPU only, set:

```python
DEVICE = "cpu"
AMP = False
```

## Outputs

Training creates:

- `checkpoints/epoch_*.pth`
- `checkpoints/best.pth`
- `runs/` (TensorBoard logs)
- `plots/training_history.csv`
- `plots/loss_curve.png`
- `plots/val_metrics_curve.png`
- `plots/lr_curve.png`

Evaluation creates:

- `plots/confusion_matrix.png`
- `plots/confusion_matrix_normalized.png`
- `plots/misclassification_heatmap.png`
- `plots/per_class_metrics.png`
- `plots/per_class_metrics.csv`
- `plots/top_misclassified_pairs.csv` (if applicable)

## TensorBoard

```powershell
uv run tensorboard --logdir runs
```

Then open the local TensorBoard URL shown in terminal.

## Troubleshooting

- `Checkpoint not found: checkpoints/best.pth`
  - Train first: `uv run python train.py`.
- CUDA not used
  - Confirm GPU PyTorch build is installed and `torch.cuda.is_available()` is `True`.
  - Check `DEVICE` in `config.py`.
- Data loading issues
  - Verify `NEU-CLS-64` exists and has one subfolder per class.
# SteelViT Defect Classification

This project trains and evaluates a steel surface defect classifier on the `NEU-CLS-64` dataset.

## What Kind of Model Is This?

This is a **supervised multiclass image classification model** with a **hybrid CNN + Vision Transformer (ViT)** architecture.

- Task: classify each image into 1 of 9 defect classes (`cr`, `gg`, `in`, `pa`, `ps`, `rp`, `rs`, `sc`, `sp`)
- Input: grayscale image `[1, 64, 64]`
- Output: class logits `[9]`
- Learning setup: supervised learning from folder-based labels

## Model Architecture (SteelViT)

The model is implemented in `model.py` and follows this pipeline:

1. **CNN stem** for local texture extraction  
   `1x64x64 -> 32x64x64 -> 64x32x32 -> 128x16x16`
2. **Patch embedding**  
   `128x16x16 -> 192x8x8`, flattened into 64 tokens
3. **Transformer encoder**  
   6 encoder blocks, 3 attention heads, MLP ratio 4.0
4. **Classification head**  
   CLS token -> LayerNorm -> Linear -> 9 classes

This is not a pure CNN and not a pure ViT. It combines CNN locality (good for defect texture) with Transformer global context modeling.

## Training Details

- Optimizer: AdamW
- LR schedule: warmup + cosine decay
- Regularization: label smoothing, MixUp, CutMix, stochastic depth
- Precision: AMP on CUDA
- Imbalance handling: class-weighted loss (configurable), optional weighted sampler

## Run

Train:

```powershell
uv run python train.py
```

Evaluate:

```powershell
uv run python evaluate.py
```

Plots and metrics are saved under `plots/`.
