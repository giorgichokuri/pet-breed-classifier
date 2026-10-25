"""Model definitions: a small CNN trained from scratch (baseline) and pretrained backbones."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES

MODEL_NAMES = ["simple_cnn", "resnet18", "efficientnet_b0", "mobilenet_v3_large"]


class SimpleCNN(nn.Module):
    """Baseline trained from scratch: 4 conv blocks -> global average pooling -> classifier.

    Each block: Conv(3x3) -> BatchNorm -> ReLU -> MaxPool(2), so the image gets smaller
    and the number of channels (learned features) grows: 3 -> 32 -> 64 -> 128 -> 256.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module:
    """Create a model and replace its last layer with a new 37-class head.

    freeze_backbone=True trains only the new head ("linear probe"): fast, but the
    features stay exactly as learned on ImageNet.
    """
    if name == "simple_cnn":
        return SimpleCNN(num_classes)

    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        head = m.classifier
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        head = m.classifier
    else:
        raise ValueError(f"unknown model {name}; choose from {MODEL_NAMES}")

    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return m


def head_parameters(model: nn.Module, name: str):
    """Parameters of the new classification head (they get a higher learning rate)."""
    if name == "resnet18":
        return list(model.fc.parameters())
    if name in ("efficientnet_b0", "mobilenet_v3_large"):
        return list(model.classifier.parameters())
    return list(model.parameters())


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
