"""
evaluate.py
Generates detailed evaluation metrics for a trained EcoSort AI model:
- Overall accuracy
- Per-class precision / recall / F1
- Confusion matrix plot

Usage:
    python src/evaluate.py --config configs/config.yaml --checkpoint models/best_model.pth
"""

import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from model import build_model


def main(config_path: str, checkpoint_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    _, _, test_loader, _ = get_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["data"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["train"]["seed"],
    )

    model = build_model(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("EcoSort AI — Confusion Matrix")
    plt.tight_layout()

    out_path = os.path.join(cfg["train"]["checkpoint_dir"], "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nConfusion matrix saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
