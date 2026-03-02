"""
train_resnet.py  —  LSAformer + ResNet-18  |  NEU-CLS-64

Default clean training pipeline:
  - CrossEntropyLoss with class weights + label smoothing
  - Standard augmentation (flip, crop, rotate, jitter, erasing)
  - WeightedRandomSampler for class imbalance
  - Two-phase training: backbone frozen warmup → full fine-tune

Usage:
    python train_resnet.py --data_root "E:/VisionT/NEU-CLS-64" --in_channels 1 --model_size tiny
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF
import random

from model_resnet_freq import lsaformer_resnet_tiny, lsaformer_resnet_small, lsaformer_resnet_base


CLASS_NAMES  = ['cr', 'gg', 'in', 'pa', 'ps', 'rp', 'rs', 'sc', 'sp']
CLASS_COUNTS = [1209, 296, 774, 1148, 797, 200, 1589, 773, 438]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Augmentations
# ─────────────────────────────────────────────────────────────────────────────

class GrayscaleJitter:
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast   = contrast

    def __call__(self, img):
        if random.random() > 0.5:
            img = TF.adjust_brightness(img, 1 + random.uniform(-self.brightness, self.brightness))
        if random.random() > 0.5:
            img = TF.adjust_contrast(img, 1 + random.uniform(-self.contrast, self.contrast))
        return img


def build_transforms(train=True, image_size=64, in_channels=1):
    to_gray = [transforms.Grayscale(num_output_channels=1)] if in_channels == 1 else []
    mean    = (0.5,) * in_channels
    std     = (0.5,) * in_channels

    if train:
        return transforms.Compose([
            *to_gray,
            transforms.Resize((image_size + 8, image_size + 8)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            GrayscaleJitter(brightness=0.2, contrast=0.2),
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


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(data_root, batch_size=32, val_split=0.15, test_split=0.10,
                      image_size=64, in_channels=1):

    train_full = datasets.ImageFolder(
        data_root,
        transform=build_transforms(train=True, image_size=image_size, in_channels=in_channels),
    )
    eval_full = datasets.ImageFolder(
        data_root,
        transform=build_transforms(train=False, image_size=image_size, in_channels=in_channels),
    )

    n       = len(train_full)
    n_val   = int(n * val_split)
    n_test  = int(n * test_split)
    n_train = n - n_val - n_test

    torch.manual_seed(42)
    idx       = torch.randperm(n).tolist()
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(eval_full,  val_idx)
    test_ds  = Subset(eval_full,  test_idx)

    # Standard weighted sampler — 1/count per class, no extra boosts
    counts         = torch.tensor(CLASS_COUNTS, dtype=torch.float)
    weights        = 1.0 / counts
    weights        = weights / weights.sum() * len(CLASS_NAMES)

    targets        = [train_full.targets[i] for i in train_idx]
    sample_weights = torch.tensor([weights[t].item() for t in targets])
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Splits  — Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"Classes — {train_full.classes}")
    return train_loader, val_loader, test_loader, train_full.classes


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Loss
# ─────────────────────────────────────────────────────────────────────────────

def build_criterion(device):
    counts  = torch.tensor(CLASS_COUNTS, dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(CLASS_NAMES)
    return nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def step(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = val_acc
            self.counter    = 0


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

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
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def per_class_accuracy(preds, labels, class_names):
    correct_c = defaultdict(int)
    total_c   = defaultdict(int)
    for p, l in zip(preds, labels):
        total_c[l] += 1
        if p == l:
            correct_c[l] += 1
    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        acc  = correct_c[i] / max(total_c[i], 1) * 100
        flag = ' ⚠' if acc < 90 else (' ★' if acc >= 99 else '')
        print(f"  {name:>4s}  ({total_c[i]:4d} samples): {acc:.1f}%{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Save helper
# ─────────────────────────────────────────────────────────────────────────────

def save_best(model, epoch, val_acc, class_names, args, out_dir):
    torch.save({
        'epoch':            epoch,
        'model_state_dict': model.state_dict(),
        'val_acc':          val_acc,
        'class_names':      class_names,
        'in_channels':      args.in_channels,
        'model_size':       args.model_size,
    }, out_dir / 'best_model_resnet.pth')
    print(f"    ✓ Best saved (val_acc={val_acc*100:.2f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        in_channels=args.in_channels,
    )

    model_fn = {
        'tiny':  lsaformer_resnet_tiny,
        'small': lsaformer_resnet_small,
        'base':  lsaformer_resnet_base,
    }
    model = model_fn[args.model_size](
        num_classes=len(class_names),
        in_channels=args.in_channels,
        pretrained=args.pretrained,
        freeze_backbone=True,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model   : LSAformerResNet-{args.model_size} | "
          f"Total: {total:,} | Trainable: {trainable:,}")
    print(f"Loss    : CrossEntropyLoss (class-weighted, label_smoothing=0.1)")

    criterion = build_criterion(device)
    scaler    = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history      = []

    # ── Phase 1: Warmup — backbone frozen ────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"PHASE 1: Warmup ({args.warmup_epochs} epochs, backbone frozen)")
    print(f"{'='*55}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(args.warmup_epochs, 5))

    for epoch in range(1, args.warmup_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc          = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _  = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  [{epoch:3d}/{args.warmup_epochs}] "
              f"train {tr_acc*100:.2f}% | val {val_acc*100:.2f}% | "
              f"loss {val_loss:.4f} | {time.time()-t0:.1f}s")
        history.append({'epoch': epoch, 'phase': 1,
                        'train_loss': tr_loss, 'train_acc': tr_acc,
                        'val_loss': val_loss,  'val_acc': val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best(model, epoch, val_acc, class_names, args, out_dir)

    # ── Phase 2: Full fine-tuning — backbone unfrozen ─────────────────────────
    print(f"\n{'='*55}")
    print(f"PHASE 2: Fine-tuning ({args.epochs} epochs, backbone unfrozen)")
    print(f"{'='*55}")

    model.unfreeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params now: {trainable:,}")

    optimizer  = AdamW(
        model.param_groups(lr_backbone=args.lr / 10, lr_transformer=args.lr),
        weight_decay=args.weight_decay,
    )
    scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs // 3, T_mult=2)
    early_stop = EarlyStopping(patience=args.patience)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc          = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _  = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        global_ep = args.warmup_epochs + epoch
        lr_now    = optimizer.param_groups[1]['lr']

        print(f"  [{epoch:3d}/{args.epochs}] "
              f"train {tr_acc*100:.2f}% | val {val_acc*100:.2f}% | "
              f"loss {val_loss:.4f} | lr {lr_now:.2e} | {time.time()-t0:.1f}s")

        history.append({'epoch': global_ep, 'phase': 2,
                        'train_loss': tr_loss, 'train_acc': tr_acc,
                        'val_loss': val_loss,  'val_acc': val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best(model, global_ep, val_acc, class_names, args, out_dir)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), out_dir / f'resnet_ckpt_ep{global_ep}.pth')

        early_stop.step(val_acc)
        if early_stop.stop:
            print(f"\n  Early stopping at epoch {global_ep}.")
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n─── Test evaluation (best model) ───")
    ckpt = torch.load(out_dir / 'best_model_resnet.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_acc, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, device)

    print(f"\nTest accuracy : {test_acc*100:.2f}%")
    print(f"Test loss     : {test_loss:.4f}")
    per_class_accuracy(test_preds, test_labels, class_names)

    with open(out_dir / 'training_history_resnet.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy : {best_val_acc*100:.2f}%")
    print(f"Saved to          : {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSAformer+ResNet-18 on NEU-CLS-64')

    parser.add_argument('--data_root',     type=str,   default='E:/VisionT/NEU-CLS-64')
    parser.add_argument('--image_size',    type=int,   default=64)
    parser.add_argument('--in_channels',   type=int,   default=1)
    parser.add_argument('--model_size',    type=str,   default='tiny',
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--pretrained',    action='store_true', default=True)
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--warmup_epochs', type=int,   default=10)
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-2)
    parser.add_argument('--patience',      type=int,   default=15)
    parser.add_argument('--output_dir',    type=str,   default='./checkpoints')

    args = parser.parse_args()
    main(args)