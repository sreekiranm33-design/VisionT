import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config as cfg
from dataset import create_dataloaders
from model import SteelViT


def resolve_device():
    if cfg.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_normalized_confusion_matrix(cm, class_names, output_path):
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(cm, row_sums, where=row_sums != 0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        normalized_cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_misclassification_heatmap(cm, class_names, output_path):
    misclassified = cm.copy().astype(float)
    np.fill_diagonal(misclassified, 0.0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        misclassified,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Misclassification Heatmap (Off-Diagonal Counts)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_per_class_bars(class_names, precision, recall, f1, accuracy, output_path):
    x = np.arange(len(class_names))
    width = 0.2
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, precision, width, label="precision")
    plt.bar(x - 0.5 * width, recall, width, label="recall")
    plt.bar(x + 0.5 * width, f1, width, label="f1")
    plt.bar(x + 1.5 * width, accuracy, width, label="accuracy")
    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Per-Class Metrics")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def top_misclassified_pairs(cm, class_names, top_k=5):
    pairs = []
    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count > 0:
                pairs.append((count, class_names[true_idx], class_names[pred_idx]))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return pairs[:top_k]


def main():
    device = resolve_device()
    use_amp = cfg.AMP and device.type == "cuda"

    _, _, test_loader, class_names = create_dataloaders()
    checkpoint_path = cfg.CHECKPOINT_DIR / "best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = SteelViT(num_classes=cfg.NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)

            preds = logits.argmax(dim=1)
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    cm = confusion_matrix(all_targets, all_preds)
    accuracy = accuracy_score(all_targets, all_preds)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(f"Total test accuracy: {accuracy:.4f}")
    print(
        f"Macro precision: {macro_precision:.4f} | "
        f"Macro recall: {macro_recall:.4f} | "
        f"Macro F1: {macro_f1:.4f}"
    )
    print("\nPer-class accuracy:")

    class_totals = cm.sum(axis=1)
    class_correct = np.diag(cm)
    for idx, class_name in enumerate(class_names):
        total = int(class_totals[idx])
        class_acc = (class_correct[idx] / total) if total > 0 else 0.0
        print(f"  {class_name}: {class_acc:.4f} ({int(class_correct[idx])}/{total})")

    print("\nClassification report:")
    print(report)

    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )
    class_totals = cm.sum(axis=1, keepdims=False).astype(float)
    class_accuracy = np.zeros(len(class_names), dtype=float)
    np.divide(
        np.diag(cm).astype(float),
        class_totals,
        out=class_accuracy,
        where=class_totals != 0,
    )

    plots_dir = cfg.PROJECT_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    confusion_path = plots_dir / "confusion_matrix.png"
    normalized_confusion_path = plots_dir / "confusion_matrix_normalized.png"
    misclassification_path = plots_dir / "misclassification_heatmap.png"
    per_class_path = plots_dir / "per_class_metrics.png"

    save_confusion_matrix(cm, class_names, confusion_path)
    save_normalized_confusion_matrix(cm, class_names, normalized_confusion_path)
    save_misclassification_heatmap(cm, class_names, misclassification_path)
    save_per_class_bars(
        class_names=class_names,
        precision=class_precision,
        recall=class_recall,
        f1=class_f1,
        accuracy=class_accuracy,
        output_path=per_class_path,
    )

    per_class_df = pd.DataFrame(
        {
            "class": class_names,
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
            "accuracy": class_accuracy,
            "support": cm.sum(axis=1),
        }
    )
    per_class_csv = plots_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    print(f"Saved evaluation plots and metrics to: {plots_dir}")

    top_pairs = top_misclassified_pairs(cm, class_names, top_k=5)
    print("\nTop misclassified pairs:")
    if not top_pairs:
        print("  None.")
    else:
        for count, true_label, pred_label in top_pairs:
            print(f"  {true_label} -> {pred_label}: {count}")

    if top_pairs:
        pair_df = pd.DataFrame(top_pairs, columns=["count", "true_label", "pred_label"])
        pair_df.to_csv(plots_dir / "top_misclassified_pairs.csv", index=False)


if __name__ == "__main__":
    main()
