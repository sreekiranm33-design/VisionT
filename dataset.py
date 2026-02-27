import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

import config as cfg


def build_transforms(is_train):
    if is_train:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandAugment(num_ops=cfg.RANDAUG_N, magnitude=cfg.RANDAUG_M),
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.NORM_MEAN, std=cfg.NORM_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.NORM_MEAN, std=cfg.NORM_STD),
        ]
    )


def build_datasets(data_dir=cfg.DATA_DIR, random_seed=cfg.RANDOM_SEED):
    base_dataset = datasets.ImageFolder(root=str(data_dir))
    indices = np.arange(len(base_dataset))
    targets = np.array(base_dataset.targets)

    train_idx, temp_idx, _, temp_targets = train_test_split(
        indices,
        targets,
        test_size=(1.0 - cfg.TRAIN_SPLIT),
        stratify=targets,
        random_state=random_seed,
    )

    val_ratio = cfg.VAL_SPLIT / (cfg.VAL_SPLIT + cfg.TEST_SPLIT)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_ratio),
        stratify=temp_targets,
        random_state=random_seed,
    )

    train_dataset_full = datasets.ImageFolder(
        root=str(data_dir), transform=build_transforms(is_train=True)
    )
    eval_dataset_full = datasets.ImageFolder(
        root=str(data_dir), transform=build_transforms(is_train=False)
    )

    train_dataset = Subset(train_dataset_full, train_idx.tolist())
    val_dataset = Subset(eval_dataset_full, val_idx.tolist())
    test_dataset = Subset(eval_dataset_full, test_idx.tolist())

    train_targets = targets[train_idx]
    train_class_counts = np.bincount(train_targets, minlength=cfg.NUM_CLASSES)

    metadata = {
        "train_targets": train_targets.tolist(),
        "train_class_counts": train_class_counts.tolist(),
    }

    return train_dataset, val_dataset, test_dataset, base_dataset.classes, metadata


def create_dataloaders(
    batch_size=cfg.BATCH_SIZE,
    num_workers=cfg.NUM_WORKERS,
    data_dir=cfg.DATA_DIR,
    return_metadata=False,
):
    train_dataset, val_dataset, test_dataset, class_names, metadata = build_datasets(
        data_dir=data_dir
    )

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    sampler = None
    shuffle = True

    if cfg.USE_WEIGHTED_SAMPLER:
        train_targets = np.array(metadata["train_targets"], dtype=np.int64)
        class_counts = np.array(metadata["train_class_counts"], dtype=np.float64)
        class_counts = np.maximum(class_counts, 1.0)
        class_sampling_weights = 1.0 / class_counts
        sample_weights = class_sampling_weights[train_targets]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    if return_metadata:
        return train_loader, val_loader, test_loader, class_names, metadata
    return train_loader, val_loader, test_loader, class_names
