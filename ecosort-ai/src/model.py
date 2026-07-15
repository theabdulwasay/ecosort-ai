"""
model.py
Defines the transfer-learning model architecture for EcoSort AI.
"""

import torch.nn as nn
from torchvision import models


def build_model(backbone: str, num_classes: int, pretrained: bool = True,
                 dropout: float = 0.3):
    """
    Builds a classification model using a pretrained CNN backbone
    with a custom classification head.
    """
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        # expose the final conv layer name for Grad-CAM
        net.target_layer_name = "layer4"

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        net.target_layer_name = "features"

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return net


def freeze_backbone(model, backbone: str):
    """Freezes all layers except the classification head (for warm-up epochs)."""
    for name, param in model.named_parameters():
        if backbone == "resnet18" and "fc" not in name:
            param.requires_grad = False
        elif backbone == "efficientnet_b0" and "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreezes all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
