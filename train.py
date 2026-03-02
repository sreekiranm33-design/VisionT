"""
train.py  —  Training pipeline for LSAformer on NEU-CLS-64 (9 classes)

Usage:
    python train.py --data_root E:/VisionT/NEU-CLS-64 --epochs 100 --batch_size 64

Directory structure expected:
    NEU-CLS-64/
        cr/  gg/  in/  pa/  ps/  rp/  rs/  sc/  sp/
"""

import argparse
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from torchvision.transforms import RandAugment

from model import lsaformer_small, lsaformer_tiny, lsaformer_base


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ['cr', 'gg', 'in', 'pa', 'ps', 'rp', 'rs', 'sc', 'sp']
CLASS_COUNTS = [1209, 296, 774, 1148, 797, 200, 1589, 773, 438]  # from your dataset


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def build_transforms(train=True, image_size=64, in_channels=3):
    if in_channels == 1:
        mean, std = (0.5,), (0.5,)
        to_gray = [transforms.Grayscale(num_output_channels=1)]
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        to_gray = []

    if train:
        return transforms.Compose([
            *to_gray,
            transforms.Resize((image_size + 8, image_size + 8)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.2, contrast=0.2) if in_channels == 3
                else transforms.RandomGrayscale(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            *to_gray,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def build_dataloaders(data_root, batch_size=64, val_split=0.15, test_split=0.10,
                      image_size=64, in_channels=3, num_workers=0):
    """
    Splits the flat per-class folders into train/val/test.
    Uses WeightedRandomSampler to compensate for class imbalance.
    """
    import numpy as np
    from torch.utils.data import Subset, random_split

    full_dataset = datasets.ImageFolder(
        data_root,
        transform=build_transforms(train=True, image_size=image_size, in_channels=in_channels)
    )

    n = len(full_dataset)
    n_val  = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    torch.manual_seed(42)
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])

    # Overwrite transform for val/test (no augmentation)
    val_ds.dataset  = datasets.ImageFolder(
        data_root, transform=build_transforms(train=False, image_size=image_size, in_channels=in_channels))
    test_ds.dataset = datasets.ImageFolder(
        data_root, transform=build_transforms(train=False, image_size=image_size, in_channels=in_channels))

    # Weighted sampler for class balance
    targets = [full_dataset.targets[i] for i in train_ds.indices]
    counts  = torch.tensor(CLASS_COUNTS, dtype=torch.float)
    weights = 1.0 / counts
    sample_weights = torch.tensor([weights[t] for t in targets])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Dataset splits — Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"Classes: {full_dataset.classes}")

    return train_loader, val_loader, test_loader, full_dataset.classes


# ─────────────────────────────────────────────────────────────────────────────
# Loss — Label Smoothing + Class Weights
# ─────────────────────────────────────────────────────────────────────────────

def build_criterion(device):
    counts = torch.tensor(CLASS_COUNTS, dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(CLASS_COUNTS)
    return nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Train / Eval loops
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Per-class accuracy
# ─────────────────────────────────────────────────────────────────────────────

def per_class_accuracy(preds, labels, class_names):
    from collections import defaultdict
    correct_per_class = defaultdict(int)
    total_per_class   = defaultdict(int)

    for p, l in zip(preds, labels):
        total_per_class[l] += 1
        if p == l:
            correct_per_class[l] += 1

    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        acc = correct_per_class[i] / max(total_per_class[i], 1) * 100
        print(f"  {name:>4s}  ({total_per_class[i]:4d} samples): {acc:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_root, batch_size=args.batch_size,
        image_size=args.image_size, in_channels=args.in_channels
    )

    # Model
    model_fn = {'tiny': lsaformer_tiny, 'small': lsaformer_small, 'base': lsaformer_base}
    model = model_fn[args.model_size](
        num_classes=len(class_names),
        image_size=args.image_size,
        in_channels=args.in_channels
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: LSAformer-{args.model_size} | Parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = build_criterion(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs // 3, T_mult=2)
    scaler    = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss,   val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
              f"Val loss {val_loss:.4f} acc {val_acc*100:.2f}% | "
              f"LR {lr_now:.6f} | {elapsed:.1f}s")

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc
        })

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, out_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_acc={val_acc*100:.2f}%)")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), out_dir / f'checkpoint_epoch{epoch}.pth')

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n─── Loading best model for final test evaluation ───")
    ckpt = torch.load(out_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc*100:.2f}%  |  Test loss: {test_loss:.4f}")
    per_class_accuracy(test_preds, test_labels, class_names)

    # Save history
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Outputs saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSAformer on NEU-CLS-64')

    # Data
    parser.add_argument('--data_root',   type=str, default='E:/VisionT/NEU-CLS-64')
    parser.add_argument('--image_size',  type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=3,
                        help='1=grayscale, 3=RGB (NEU-CLS images are grayscale-like; try 1)')

    # Model
    parser.add_argument('--model_size',  type=str, default='small',
                        choices=['tiny', 'small', 'base'])

    # Training
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-2)

    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints')

    args = parser.parse_args()
    main(args)