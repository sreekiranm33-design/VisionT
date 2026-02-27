import math

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if warmup_epochs > 0 and current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def mixup_data(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, float(lam)


def _rand_bbox(size, lam):
    _, _, height, width = size
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return int(bbx1), int(bby1), int(bbx2), int(bby2)


def cutmix_data(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)

    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    box_area = (bbx2 - bbx1) * (bby2 - bby1)
    lam = 1.0 - box_area / float(x.size(-1) * x.size(-2))
    return mixed_x, y_a, y_b, float(lam)


def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    return total_params
