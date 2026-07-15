"""
gradcam.py
Generates Grad-CAM heatmaps to visualize which regions of an image
influenced the model's prediction — key for explainability in interviews.

Usage:
    python src/gradcam.py --checkpoint models/best_model.pth --image path/to/image.jpg
"""

import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

from model import build_model


def load_image(image_path: str, img_size: int):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    # unnormalized version for overlay visualization
    raw = np.array(img.resize((img_size, img_size))) / 255.0
    return tensor, raw


def main(checkpoint_path: str, image_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    class_names = checkpoint["class_names"]

    model = build_model(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_tensor, raw_image = load_image(image_path, cfg["data"]["img_size"])
    input_tensor = input_tensor.to(device)

    # target layer for CAM (last conv block)
    if cfg["model"]["backbone"] == "resnet18":
        target_layers = [model.layer4[-1]]
    else:
        target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(raw_image.astype(np.float32), grayscale_cam, use_rgb=True)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

    print(f"Prediction: {class_names[pred_idx]} ({confidence*100:.1f}% confidence)")

    plt.figure(figsize=(6, 6))
    plt.imshow(visualization)
    plt.title(f"Grad-CAM: {class_names[pred_idx]} ({confidence*100:.1f}%)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("gradcam_output.png", dpi=150)
    print("Saved visualization to gradcam_output.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.image)
