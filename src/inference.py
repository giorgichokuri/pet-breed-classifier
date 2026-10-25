"""Lightweight inference with ONNX Runtime (no PyTorch needed when serving).

Preprocessing here must match the evaluation transform used in training:
resize the short side to 256 -> center crop 224 -> scale to [0, 1] -> ImageNet normalization.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, RESIZE_SIZE, species_of


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    scale = RESIZE_SIZE / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return x.transpose(2, 0, 1)[None]          # HWC -> NCHW with batch dimension


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Predictor:
    """Loads model.onnx (or model_int8.onnx) and the class list saved next to it."""

    def __init__(self, model_path: str | Path, classes_path: str | Path | None = None, threads: int = 0):
        model_path = Path(model_path)
        classes_path = Path(classes_path) if classes_path else model_path.parent / "classes.json"
        self.classes = json.loads(classes_path.read_text())
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img: Image.Image, top_k: int = 5) -> dict:
        t0 = time.perf_counter()
        logits = self.session.run(None, {self.input_name: preprocess(img)})[0]
        probs = softmax(logits)[0]
        order = np.argsort(-probs)[:top_k]
        top = [{"breed": self.classes[i], "probability": round(float(probs[i]), 4)} for i in order]
        return {
            "top_k": top,
            "breed": top[0]["breed"],
            "species": species_of(top[0]["breed"]),
            "confidence": top[0]["probability"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
