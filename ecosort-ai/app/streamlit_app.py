"""
streamlit_app.py
Interactive web demo for EcoSort AI — upload an image and get a live
waste-classification prediction with confidence scores and Grad-CAM overlay.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import build_model

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="centered")

st.title("♻️ EcoSort AI")
st.caption("Intelligent Waste Classification — upload a photo of an item to classify it as "
           "cardboard, glass, metal, paper, plastic, or trash.")

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
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
    return model, class_names, cfg, device


def predict(model, image, class_names, cfg, device):
    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    return probs


if not os.path.exists(CHECKPOINT_PATH):
    st.warning(
        "⚠️ No trained model checkpoint found at `models/best_model.pth`.\n\n"
        "Train one first with:\n```\npython src/train.py --config configs/config.yaml\n```"
    )
else:
    model, class_names, cfg, device = load_model()
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded image", use_column_width=True)

        probs = predict(model, image, class_names, cfg, device)
        top_idx = int(np.argmax(probs))

        with col2:
            st.subheader(f"Prediction: **{class_names[top_idx].capitalize()}**")
            st.write(f"Confidence: {probs[top_idx]*100:.1f}%")
            st.progress(float(probs[top_idx]))

            st.markdown("##### All class probabilities")
            for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
                st.write(f"{name.capitalize():12s} {p*100:5.1f}%")
                st.progress(float(p))

st.divider()
st.caption("Built with PyTorch + Streamlit | EcoSort AI Portfolio Project")
