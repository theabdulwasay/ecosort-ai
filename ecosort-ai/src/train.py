"""
train.py
Training loop for EcoSort AI: warm-up (frozen backbone) -> full fine-tuning,
with checkpointing, early stopping, and metric logging.

Usage:
    python src/train.py --config configs/config.yaml
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model, freeze_backbone, unfreeze_all


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["train"]["seed"],
    )
    print(f"Classes found: {class_names}")

    model = build_model(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(cfg["train"]["epochs"]):
        # --- Warm-up phase: freeze backbone for first N epochs ---
        if epoch == 0:
            freeze_backbone(model, cfg["model"]["backbone"])
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=cfg["train"]["lr_head"],
                               weight_decay=cfg["train"]["weight_decay"])
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])

        if epoch == cfg["train"]["warmup_epochs"]:
            print("Unfreezing backbone for full fine-tuning...")
            unfreeze_all(model)
            optimizer = AdamW(model.parameters(),
                               lr=cfg["train"]["lr_full"],
                               weight_decay=cfg["train"]["weight_decay"])
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"] - epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # --- Checkpointing & early stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "config": cfg,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    # Final test evaluation using best checkpoint
    checkpoint = torch.load(os.path.join(cfg["train"]["checkpoint_dir"], "best_model.pth"),
                             map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
