import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from dataset import create_dataloaders
from model import SteelViT
from utils import (
    count_parameters,
    cutmix_data,
    get_cosine_schedule_with_warmup,
    mixup_cutmix_criterion,
    mixup_data,
)


def compute_class_weights(class_counts, power=1.0):
    counts = np.array(class_counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    inv = np.power(1.0 / counts, power)
    weights = inv / np.mean(inv)
    return torch.tensor(weights, dtype=torch.float32)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device():
    if cfg.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, dataloader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def main():
    set_random_seed(cfg.RANDOM_SEED)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, class_names, metadata = create_dataloaders(
        return_metadata=True
    )
    device = resolve_device()
    use_amp = cfg.AMP and device.type == "cuda"

    model = SteelViT(num_classes=cfg.NUM_CLASSES).to(device)
    count_parameters(model)

    # Dataloader/model wiring sanity check required by the PRD.
    sanity_images, _ = next(iter(train_loader))
    sanity_images = sanity_images.to(device, non_blocking=True)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=use_amp):
        sanity_logits = model(sanity_images[:2])
    print(f"Sanity check output shape: {tuple(sanity_logits.shape)}")

    class_weights = None
    if cfg.USE_CLASS_WEIGHTED_LOSS:
        class_weights = compute_class_weights(
            metadata["train_class_counts"], power=cfg.CLASS_WEIGHT_POWER
        ).to(device)
        print(f"Train class counts: {metadata['train_class_counts']}")
        print(f"Loss class weights: {[round(v, 4) for v in class_weights.tolist()]}")

    train_criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.LABEL_SMOOTHING,
        weight=class_weights,
    )
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        betas=cfg.BETAS,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    total_steps = cfg.NUM_EPOCHS * len(train_loader)
    warmup_steps = cfg.WARMUP_EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_epochs=warmup_steps,
        total_epochs=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(log_dir=str(cfg.LOG_DIR))
    plots_dir = cfg.PROJECT_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    global_step = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS}", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if random.random() < 0.5:
                mixed_images, y_a, y_b, lam = mixup_data(images, labels, cfg.MIXUP_ALPHA)
                aug_name = "mixup"
            else:
                mixed_images, y_a, y_b, lam = cutmix_data(images, labels, cfg.CUTMIX_ALPHA)
                aug_name = "cutmix"

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(mixed_images)
                loss = mixup_cutmix_criterion(train_criterion, logits, y_a, y_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            global_step += 1

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}", aug=aug_name)

        train_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar("train/loss_epoch", train_loss, epoch)

        val_loss, val_acc, val_f1 = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=eval_criterion,
            device=device,
            use_amp=use_amp,
        )
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.add_scalar("val/f1_macro", val_f1, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_f1,
            "val_acc": val_acc,
            "class_names": class_names,
        }

        if epoch % cfg.SAVE_EVERY == 0:
            torch.save(checkpoint, cfg.CHECKPOINT_DIR / f"epoch_{epoch}.pth")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(checkpoint, cfg.CHECKPOINT_DIR / "best.pth")

    writer.close()
    history_df = pd.DataFrame(history)
    history_csv = plots_dir / "training_history.csv"
    history_df.to_csv(history_csv, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["val_acc"], label="val_acc")
    plt.plot(history_df["epoch"], history_df["val_f1"], label="val_f1_macro")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Accuracy and Macro F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "val_metrics_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["lr"], label="lr")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "lr_curve.png", dpi=200)
    plt.close()

    print(f"Saved training plots and history to: {plots_dir}")
    print(f"Training complete. Best val_f1={best_val_f1:.4f}")


if __name__ == "__main__":
    main()
