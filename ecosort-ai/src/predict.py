"""
predict.py
Simple CLI inference for a single image using a trained EcoSort AI checkpoint.

Usage:
    python src/predict.py --checkpoint models/best_model.pth --image path/to/image.jpg
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms

from model import build_model


def predict(checkpoint_path: str, image_path: str, top_k: int = 3):
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

    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    top_probs, top_idxs = probs.topk(top_k)
    results = [(class_names[idx], prob.item()) for idx, prob in zip(top_idxs, top_probs)]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    results = predict(args.checkpoint, args.image, args.top_k)
    print("\nPredictions:")
    for label, prob in results:
        print(f"  {label:12s} {prob*100:.2f}%")
