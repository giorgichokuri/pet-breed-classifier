"""Small helpers shared by the scripts."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model, model_name: str, classes, extra: dict | None = None) -> None:
    torch.save({"state_dict": model.state_dict(), "model_name": model_name,
                "classes": list(classes), **(extra or {})}, path)


def load_checkpoint(path: str | Path, device="cpu"):
    """Rebuild the model from a checkpoint. Returns (model, model_name, classes)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], num_classes=len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["model_name"], ckpt["classes"]


def write_json(path: Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2))
