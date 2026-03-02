"""
predict.py  —  Inference + Visualisation for LSAformer + ResNet-18 + Freq Branch
NEU-CLS-64 Steel Surface Defect Classification

Usage:
    # Single image prediction
    python predict.py --checkpoint checkpoints/best_model_resnet.pth --image path/to/img.bmp

    # Single image + attention map
    python predict.py --checkpoint checkpoints/best_model_resnet.pth --image path/to/img.bmp --visualize

    # Full dataset evaluation + confusion matrix
    python predict.py --checkpoint checkpoints/best_model_resnet.pth --eval_dir E:/VisionT/NEU-CLS-64

    # Show how much each branch (global/local/freq) contributed
    python predict.py --checkpoint checkpoints/best_model_resnet.pth --image path/to/img.bmp --gate_viz
"""

import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model_resnet import (
    lsaformer_resnet_tiny,
    lsaformer_resnet_small,
    lsaformer_resnet_base,
)


CLASS_NAMES = ['cr', 'gg', 'in', 'pa', 'ps', 'rp', 'rs', 'sc', 'sp']


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, model_size='tiny', in_channels=1, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_fn = {
        'tiny':  lsaformer_resnet_tiny,
        'small': lsaformer_resnet_small,
        'base':  lsaformer_resnet_base,
    }

    class_names = ckpt.get('class_names', CLASS_NAMES)
    n_classes   = len(class_names)
    in_ch       = ckpt.get('in_channels', in_channels)
    size        = ckpt.get('model_size', model_size)

    model = model_fn[size](
        num_classes=n_classes,
        in_channels=in_ch,
        pretrained=False,          # weights come from checkpoint
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded model  : LSAformerResNet-{size}")
    print(f"Epoch         : {ckpt.get('epoch', '?')}")
    print(f"Val accuracy  : {ckpt.get('val_acc', 0)*100:.2f}%")
    print(f"Classes       : {class_names}")
    return model, class_names, in_ch


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Transform
# ─────────────────────────────────────────────────────────────────────────────

def build_transform(image_size=64, in_channels=1):
    ops = []
    if in_channels == 1:
        ops.append(transforms.Grayscale(1))
        mean, std = (0.5,), (0.5,)
    else:
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

    ops += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(ops)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Single image prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(model, img_path, transform, class_names, device):
    img    = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = logits.softmax(dim=1).squeeze().cpu().numpy()

    pred_idx  = probs.argmax()
    pred_name = class_names[pred_idx]

    print(f"\nImage      : {img_path}")
    print(f"Prediction : {pred_name}  ({probs[pred_idx]*100:.1f}%)")
    print("\nAll class probabilities:")
    for name, p in zip(class_names, probs):
        bar = '█' * int(p * 40)
        mark = ' ◄' if name == pred_name else ''
        print(f"  {name:>4s}: {bar:<40s} {p*100:5.1f}%{mark}")

    return pred_name, probs


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Attention map visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_attention(model, img_path, transform, class_names, device,
                        save_path='attention_map.png'):
    img    = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    attn_maps, h_feat, w_feat = model.get_attention_maps(tensor)

    # Last layer: CLS → spatial token attention (mean over heads)
    last_attn = attn_maps[-1]                    # (1, heads, N+1, N+1)
    cls_attn  = last_attn[0, :, 0, 1:]          # (heads, N_spatial)
    cls_attn  = cls_attn.mean(0).cpu().numpy()   # (N_spatial,)
    cls_attn  = cls_attn.reshape(h_feat, w_feat)

    # Normalise
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    # Upsample to image size
    attn_t  = torch.tensor(cls_attn).unsqueeze(0).unsqueeze(0)
    img_arr = np.array(img.resize((64, 64)))
    attn_up = F.interpolate(attn_t, size=(64, 64),
                            mode='bilinear', align_corners=False).squeeze().numpy()

    # Overlay
    heat    = cm.hot(attn_up)[:, :, :3] * 255
    if img_arr.ndim == 2:
        img_rgb = np.stack([img_arr] * 3, axis=-1)
    else:
        img_rgb = img_arr
    blended = (0.5 * img_rgb + 0.5 * heat).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_arr, cmap='gray' if img_arr.ndim == 2 else None)
    axes[0].set_title('Input Image');  axes[0].axis('off')
    axes[1].imshow(attn_up, cmap='hot')
    axes[1].set_title('Attention Map'); axes[1].axis('off')
    axes[2].imshow(blended)
    axes[2].set_title('Overlay');       axes[2].axis('off')

    pred_name, probs = predict_image(model, img_path, transform, class_names, device)
    conf = probs[class_names.index(pred_name)] * 100
    fig.suptitle(f'Prediction: {pred_name}  ({conf:.1f}%)', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved attention map → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Gate weight visualisation  ← NEW (freq branch specific)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_gate_weights(model, img_path, transform, class_names, device,
                           save_path='gate_weights.png'):
    """
    Shows how much each branch (global / local / freq) contributed
    at each of the 6 LSAformer blocks for this specific image.

    Gate weights are per-token — we average over all 65 tokens (CLS + patches)
    to get one scalar per branch per block.
    """
    img    = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    # ── Register forward hooks on each block's gate (Softmax layer) ──────────
    gate_outputs = []

    def make_hook():
        def hook(module, inp, out):
            # out: (B, N, 3) — 3 branch weights per token
            gate_outputs.append(out.detach().cpu())
        return hook

    hooks = []
    for block in model.blocks:
        h = block.attn.gate.register_forward_hook(make_hook())
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = logits.softmax(dim=1).squeeze().cpu().numpy()

    for h in hooks:
        h.remove()

    pred_idx  = probs.argmax()
    pred_name = class_names[pred_idx]

    # ── Extract mean gate weights per block ───────────────────────────────────
    # Each gate_output: (1, N, 3)   N=65 (CLS + 64 patches)
    n_blocks = len(gate_outputs)
    w_global = np.zeros(n_blocks)
    w_local  = np.zeros(n_blocks)
    w_freq   = np.zeros(n_blocks)

    for i, gw in enumerate(gate_outputs):
        gw = gw.squeeze(0)          # (N, 3)
        w_global[i] = gw[:, 0].mean().item()
        w_local[i]  = gw[:, 1].mean().item()
        w_freq[i]   = gw[:, 2].mean().item()

    # ── Spatial gate map for last block (freq branch weight per patch) ────────
    last_gw    = gate_outputs[-1].squeeze(0)   # (65, 3)
    freq_map   = last_gw[1:, 2].numpy()        # spatial patches only, freq weight
    h_feat     = model.h_feat
    freq_map   = freq_map.reshape(h_feat, model.w_feat)

    # Normalise
    freq_map = (freq_map - freq_map.min()) / (freq_map.max() - freq_map.min() + 1e-8)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#0e1117')

    blocks = [f'B{i+1}' for i in range(n_blocks)]
    x      = np.arange(n_blocks)
    width  = 0.25

    # ── Subplot 1: bar chart of branch weights per block ─────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#161b22')
    b1 = ax1.bar(x - width, w_global, width, label='Global Attn',
                 color='#89b4fa', alpha=0.9)
    b2 = ax1.bar(x,          w_local,  width, label='Local Conv',
                 color='#f38ba8', alpha=0.9)
    b3 = ax1.bar(x + width,  w_freq,   width, label='Freq Branch',
                 color='#a6e3a1', alpha=0.9)
    ax1.set_xticks(x); ax1.set_xticklabels(blocks, color='white')
    ax1.set_ylabel('Mean gate weight', color='white')
    ax1.set_title('Branch Contribution per Block', color='white', pad=10)
    ax1.legend(facecolor='#1e2433', labelcolor='white', framealpha=0.8)
    ax1.set_ylim(0, 0.7)
    ax1.tick_params(colors='white')
    ax1.spines[:].set_color('#2a2e3a')
    for spine in ax1.spines.values():
        spine.set_color('#2a2e3a')
    ax1.yaxis.label.set_color('white')

    # ── Subplot 2: stacked area chart ────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor('#161b22')
    ax2.stackplot(x, w_global, w_local, w_freq,
                  labels=['Global', 'Local', 'Freq'],
                  colors=['#89b4fa', '#f38ba8', '#a6e3a1'], alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(blocks, color='white')
    ax2.set_ylabel('Stacked weight (sum=1)', color='white')
    ax2.set_title('Branch Mix Across Blocks', color='white', pad=10)
    ax2.legend(loc='upper right', facecolor='#1e2433', labelcolor='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#2a2e3a')

    # ── Subplot 3: freq weight spatial map (last block) ───────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    im  = ax3.imshow(freq_map, cmap='Greens', interpolation='nearest')
    ax3.set_title('Freq Branch Weight Map\n(Block 6 — spatial tokens)',
                  color='white', pad=10)
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')

    # ── Subplot 4: per-block summary text ────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor('#161b22')
    ax4.axis('off')

    lines = [
        f"Image    : {Path(img_path).name}",
        f"Predicted: {pred_name}  ({probs[pred_idx]*100:.1f}%)",
        "",
        f"{'Block':<6} {'Global':>8} {'Local':>8} {'Freq':>8}",
        "─" * 34,
    ]
    for i in range(n_blocks):
        dominant = ['Global', 'Local ', 'Freq  '][
            np.argmax([w_global[i], w_local[i], w_freq[i]])]
        lines.append(
            f"  B{i+1}   {w_global[i]:>8.3f} {w_local[i]:>8.3f} "
            f"{w_freq[i]:>8.3f}  ← {dominant}"
        )
    lines += [
        "─" * 34,
        f"{'Mean':<6} {w_global.mean():>8.3f} {w_local.mean():>8.3f} "
        f"{w_freq.mean():>8.3f}",
    ]

    ax4.text(0.05, 0.95, '\n'.join(lines),
             transform=ax4.transAxes,
             fontsize=8.5, verticalalignment='top',
             fontfamily='monospace',
             color='#cdd6f4',
             bbox=dict(boxstyle='round', facecolor='#1e2433', alpha=0.8))

    fig.suptitle(
        f'LSAformer Frequency Branch Analysis  |  {pred_name}  ({probs[pred_idx]*100:.1f}%)',
        color='white', fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nSaved gate weight visualisation → {save_path}")
    plt.show()

    # Print summary to terminal
    print(f"\n{'─'*40}")
    print(f"Gate weight summary for: {pred_name}")
    print(f"{'─'*40}")
    print(f"{'Block':<8} {'Global':>8} {'Local':>8} {'Freq':>8}")
    for i in range(n_blocks):
        print(f"  B{i+1:<5}  {w_global[i]:>8.3f} {w_local[i]:>8.3f} {w_freq[i]:>8.3f}")
    print(f"{'Mean':<8} {w_global.mean():>8.3f} {w_local.mean():>8.3f} {w_freq.mean():>8.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Gate weights across entire dataset (per class average)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def gate_weights_per_class(model, eval_dir, transform, class_names, device,
                           batch_size=32, save_path='gate_per_class.png'):
    """
    Runs the full dataset and averages freq branch gate weight per class.
    Tells you which defect types the model relies on frequency features most for.
    """
    dataset = datasets.ImageFolder(eval_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)

    # Accumulate gate weights per true class
    class_gate_sums   = defaultdict(lambda: np.zeros(3))
    class_gate_counts = defaultdict(int)

    # Hook on last block's gate only (most informative)
    last_block_gate = []
    def hook(module, inp, out):
        last_block_gate.append(out.detach().cpu())

    h = model.blocks[-1].attn.gate.register_forward_hook(hook)

    model.eval()
    for imgs, labels in loader:
        imgs = imgs.to(device)
        last_block_gate.clear()
        _ = model(imgs)

        gw = last_block_gate[0]    # (B, N, 3)
        gw_mean = gw.mean(dim=1)   # (B, 3) — average over tokens

        for b_idx, lbl in enumerate(labels):
            cls_idx = lbl.item()
            class_gate_sums[cls_idx]   += gw_mean[b_idx].numpy()
            class_gate_counts[cls_idx] += 1

    h.remove()

    # Compute averages
    avg_global = np.array([class_gate_sums[i][0] / max(class_gate_counts[i], 1)
                            for i in range(len(class_names))])
    avg_local  = np.array([class_gate_sums[i][1] / max(class_gate_counts[i], 1)
                            for i in range(len(class_names))])
    avg_freq   = np.array([class_gate_sums[i][2] / max(class_gate_counts[i], 1)
                            for i in range(len(class_names))])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')

    x     = np.arange(len(class_names))
    width = 0.25

    ax.bar(x - width, avg_global, width, label='Global Attn', color='#89b4fa', alpha=0.9)
    ax.bar(x,          avg_local,  width, label='Local Conv',  color='#f38ba8', alpha=0.9)
    ax.bar(x + width,  avg_freq,   width, label='Freq Branch', color='#a6e3a1', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, color='white', fontsize=11)
    ax.set_ylabel('Mean gate weight (Block 6)', color='white')
    ax.set_title('Branch Contribution per Defect Class  |  LSAformer Freq Branch',
                 color='white', pad=12)
    ax.legend(facecolor='#1e2433', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#2a2e3a')
    ax.yaxis.label.set_color('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nSaved per-class gate analysis → {save_path}")
    plt.show()

    # Terminal summary
    print(f"\n{'─'*52}")
    print(f"{'Class':<6} {'Global':>8} {'Local':>8} {'Freq':>8}  Dominant")
    print(f"{'─'*52}")
    for i, name in enumerate(class_names):
        dominant = ['Global', 'Local ', 'Freq  '][
            np.argmax([avg_global[i], avg_local[i], avg_freq[i]])]
        print(f"{name:<6} {avg_global[i]:>8.3f} {avg_local[i]:>8.3f} "
              f"{avg_freq[i]:>8.3f}  {dominant}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Full dataset evaluation + confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_folder(model, eval_dir, transform, class_names, device, batch_size=64):
    dataset = datasets.ImageFolder(eval_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())

    print("\n" + "─" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm_arr = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('LSAformer — Confusion Matrix (NEU-CLS-64)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("Saved confusion matrix → confusion_matrix.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict + Visualise LSAformer + ResNet + Freq Branch')

    parser.add_argument('--checkpoint',   required=True,
                        help='Path to best_model_resnet.pth')
    parser.add_argument('--image',        type=str, default=None,
                        help='Single image path for prediction')
    parser.add_argument('--eval_dir',     type=str, default=None,
                        help='Dataset folder for full evaluation')
    parser.add_argument('--visualize',    action='store_true',
                        help='Show attention map for --image')
    parser.add_argument('--gate_viz',     action='store_true',
                        help='Show freq branch gate weights for --image')
    parser.add_argument('--gate_per_class', action='store_true',
                        help='Show gate weights averaged per class (needs --eval_dir)')
    parser.add_argument('--model_size',   type=str, default='tiny',
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--image_size',   type=int, default=64)
    parser.add_argument('--in_channels',  type=int, default=1)
    parser.add_argument('--batch_size',   type=int, default=64)

    args   = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, class_names, in_ch = load_model(
        args.checkpoint,
        model_size=args.model_size,
        in_channels=args.in_channels,
        device=device,
    )
    tf = build_transform(args.image_size, in_ch)

    # ── Single image modes ────────────────────────────────────────────────────
    if args.image:
        predict_image(model, args.image, tf, class_names, device)

        if args.visualize:
            visualize_attention(model, args.image, tf, class_names, device,
                                save_path='attention_map.png')

        if args.gate_viz:
            visualize_gate_weights(model, args.image, tf, class_names, device,
                                   save_path='gate_weights.png')

    # ── Dataset evaluation modes ──────────────────────────────────────────────
    if args.eval_dir:
        evaluate_folder(model, args.eval_dir, tf, class_names, device,
                        batch_size=args.batch_size)

        if args.gate_per_class:
            gate_weights_per_class(model, args.eval_dir, tf, class_names, device,
                                   batch_size=args.batch_size,
                                   save_path='gate_per_class.png')