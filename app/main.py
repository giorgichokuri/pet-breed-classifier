"""FastAPI service: upload a photo, get the top breed predictions.

    uvicorn app.main:app --reload        # then open http://localhost:8000
"""
from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from src.inference import Predictor

ROOT = Path(__file__).resolve().parent.parent
# FP32 ONNX by default: static INT8 lost too much accuracy for EfficientNet-B0 (see report, section 4).
# Set MODEL_PATH=models/model_int8.onnx to serve a quantized model instead.
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.onnx"))
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Pet Breed Classifier", description="37 cat and dog breeds (Oxford-IIIT Pet)")
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Load the model once, on first use."""
    global _predictor
    if _predictor is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, f"Model not found at {MODEL_PATH}. Run `python -m src.optimize` first.")
        _predictor = Predictor(MODEL_PATH)
    return _predictor


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "model_loaded": _predictor is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=37)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Please upload a valid image (JPG, PNG or WebP).")
    return get_predictor().predict(img, top_k)
