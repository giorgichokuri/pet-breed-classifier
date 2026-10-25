"""Fast checks that run without the real dataset or pretrained weights:  pytest -q"""
import importlib
import io
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import NUM_CLASSES
from src.data import build_transforms
from src.inference import preprocess
from src.models import MODEL_NAMES, build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_output_shape(name):
    model = build_model(name, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, NUM_CLASSES)


def test_freeze_trains_only_head():
    model = build_model("resnet18", pretrained=False, freeze_backbone=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["fc.weight", "fc.bias"]


def test_serving_preprocessing_matches_training():
    """The numpy preprocessing used by the API must match the torchvision eval transform."""
    img = Image.fromarray((np.random.default_rng(0).random((300, 420, 3)) * 255).astype("uint8"))
    _, eval_tf = build_transforms("none")
    ref = eval_tf(img).numpy()[None]
    ours = preprocess(img)
    assert ours.shape == ref.shape == (1, 3, 224, 224)
    assert np.abs(ours - ref).mean() < 0.05


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    model = build_model("simple_cnn").eval()
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), str(d / "model.onnx"),
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}}, opset_version=17, dynamo=False)
    (d / "classes.json").write_text(json.dumps([f"Breed {i}" for i in range(NUM_CLASSES)]))
    os.environ["MODEL_PATH"] = str(d / "model.onnx")
    import app.main
    importlib.reload(app.main)
    from fastapi.testclient import TestClient
    return TestClient(app.main.app)


def _jpeg():
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8")).save(buf, "JPEG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client):
    r = client.post("/predict?top_k=3", files={"file": ("cat.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert len(data["top_k"]) == 3
    assert abs(sum(p["probability"] for p in data["top_k"])) <= 1.0001
    assert data["species"] in ("cat", "dog")


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_index_page(client):
    assert "Pet Breed Classifier" in client.get("/").text
