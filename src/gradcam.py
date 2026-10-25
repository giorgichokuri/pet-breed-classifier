"""Grad-CAM: which part of the image made the model choose its answer.

Idea: take the last convolutional layer, look at how much each of its feature maps
affects the score of the predicted class (the gradient), weight the maps by that
importance and add them up. Bright regions = regions the decision relied on.

    python -m src.gradcam --ckpt experiments/resnet18_finetune/best.pt --n 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .data import load_datasets
from .evaluate import denormalize
from .utils import load_checkpoint


def target_layer(model, model_name):
    if model_name == "resnet18":
        return model.layer4
    if model_name in ("efficientnet_b0", "mobilenet_v3_large"):
        return model.features[-1]
    return model.features[-1]   # SimpleCNN


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = self.gradients = None
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x):
        """x: one image tensor (1, 3, H, W). Returns (heatmap HxW in [0, 1], predicted class)."""
        self.model.zero_grad()
        logits = self.model(x)
        cls = int(logits.argmax(1))
        logits[0, cls].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # importance of each map
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.numpy(), cls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()

    model, model_name, classes = load_checkpoint(a.ckpt, "cpu")
    cam = GradCAM(model, target_layer(model, model_name))
    _, _, test_ds, _ = load_datasets(a.data_dir, "none", fake=a.fake)
    idx = np.random.default_rng(0).choice(len(test_ds), size=min(a.n, len(test_ds)), replace=False)

    fig, axes = plt.subplots(2, len(idx), figsize=(len(idx) * 2.6, 5.6))
    axes = np.array(axes).reshape(2, -1)
    for col, i in enumerate(idx):
        x, y = test_ds[i]
        heat, pred = cam(x.unsqueeze(0))
        img = denormalize(x)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"true: {classes[y]}", fontsize=8)
        axes[1, col].imshow(img)
        axes[1, col].imshow(heat, cmap="jet", alpha=0.45)
        axes[1, col].set_title(f"pred: {classes[pred]}", fontsize=8, color="green" if pred == y else "red")
        for r in range(2):
            axes[r, col].axis("off")
    fig.tight_layout()
    out = Path(a.ckpt).parent / "gradcam.png"
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()
