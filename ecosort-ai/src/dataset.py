"""
dataset.py
Handles dataset loading, augmentation, and train/val/test splitting
for the EcoSort AI waste classification project.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(img_size: int):
    """Return train and eval transform pipelines."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def get_dataloaders(data_dir: str, img_size: int, batch_size: int,
                     val_split: float, test_split: float,
                     num_workers: int = 4, seed: int = 42):
    """
    Builds train/val/test DataLoaders from an ImageFolder-structured directory.

    Expects:
        data_dir/
            class_a/
            class_b/
            ...
    """
    train_tf, eval_tf = get_transforms(img_size)

    # Load full dataset twice with different transforms is wasteful, so we
    # load once with eval transform to compute splits, then wrap subsets.
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_names = base_dataset.classes

    total_len = len(base_dataset)
    test_len = int(total_len * test_split)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len - test_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_len, val_len, test_len], generator=generator
    )

    # Apply the right transforms to each split
    train_subset.dataset = datasets.ImageFolder(root=data_dir, transform=train_tf)
    val_subset.dataset = datasets.ImageFolder(root=data_dir, transform=eval_tf)
    test_subset.dataset = datasets.ImageFolder(root=data_dir, transform=eval_tf)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names
