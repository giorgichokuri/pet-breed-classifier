"""Optimization: export to ONNX, quantize to INT8 and compare size / speed / accuracy.

    python -m src.optimize --ckpt experiments/resnet18_finetune/best.pt

Produces in models/: model.onnx (FP32), model_int8.onnx (INT8), classes.json,
and experiments/<run>/optimization.json with the comparison table.

Why: FP32 stores every weight in 4 bytes; INT8 stores it in 1 byte. The model gets ~4x
smaller and usually faster on CPU, at the cost of a small accuracy drop that we measure.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from .config import IMG_SIZE
from .data import load_datasets
from .utils import load_checkpoint, write_json


def export_onnx(model, path: Path):
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    kwargs = dict(input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:          # older PyTorch without the `dynamo` argument
        torch.onnx.export(model, dummy, str(path), **kwargs)


class ImageReader(CalibrationDataReader):
    """Feeds a few hundred real images so the quantizer can learn the value ranges."""

    def __init__(self, dataset, n=200):
        n = min(n, len(dataset))
        self.items = iter([{"image": dataset[i][0].unsqueeze(0).numpy()} for i in range(n)])

    def get_next(self):
        return next(self.items, None)


def quantize(fp32_path: Path, int8_path: Path, calib_ds):
    prep = fp32_path.with_name("model_prep.onnx")
    quant_pre_process(str(fp32_path), str(prep))
    quantize_static(str(prep), str(int8_path), ImageReader(calib_ds),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    prep.unlink(missing_ok=True)


def latency_ms(fn, x, runs=50, warmup=5):
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 2)


def accuracy(fn, dataset, limit):
    n = min(limit, len(dataset)) if limit else len(dataset)
    correct = 0
    for i in range(n):
        x, y = dataset[i]
        correct += int(np.argmax(fn(x.unsqueeze(0).numpy())) == y)
    return round(correct / n, 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="models")
    p.add_argument("--limit", type=int, default=0, help="evaluate on the first N test images (0 = all)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    model, model_name, classes = load_checkpoint(a.ckpt, "cpu")
    _, val_ds, test_ds, _ = load_datasets(a.data_dir, "none", fake=a.fake)

    fp32, int8 = out / "model.onnx", out / "model_int8.onnx"
    export_onnx(model, fp32)
    quantize(fp32, int8, val_ds)              # calibrate on validation images, never on test
    (out / "classes.json").write_text(json.dumps(classes))

    sess32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])

    @torch.no_grad()
    def torch_fn(x):
        return model(torch.from_numpy(x)).numpy()

    def ort_fn(sess):
        return lambda x: sess.run(None, {"image": x})[0]

    x1 = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    # sanity check: the ONNX model must give the same outputs as PyTorch
    max_diff = float(np.abs(torch_fn(x1) - ort_fn(sess32)(x1)).max())

    ckpt_mb = Path(a.ckpt).stat().st_size / 1e6
    rows = [
        {"variant": "PyTorch FP32", "size_mb": round(ckpt_mb, 2), "latency_ms": latency_ms(torch_fn, x1),
         "test_acc": accuracy(torch_fn, test_ds, a.limit)},
        {"variant": "ONNX FP32", "size_mb": round(fp32.stat().st_size / 1e6, 2),
         "latency_ms": latency_ms(ort_fn(sess32), x1), "test_acc": accuracy(ort_fn(sess32), test_ds, a.limit)},
        {"variant": "ONNX INT8", "size_mb": round(int8.stat().st_size / 1e6, 2),
         "latency_ms": latency_ms(ort_fn(sess8), x1), "test_acc": accuracy(ort_fn(sess8), test_ds, a.limit)},
    ]
    result = {"model": model_name, "onnx_vs_torch_max_abs_diff": max_diff,
              "eval_images": a.limit or len(test_ds), "cpu_threads": torch.get_num_threads(), "variants": rows}
    write_json(Path(a.ckpt).parent / "optimization.json", result)

    print(f"ONNX vs PyTorch max |diff| = {max_diff:.2e}")
    print(f"{'variant':<14}{'size MB':>9}{'latency ms':>12}{'test acc':>10}")
    for r in rows:
        print(f"{r['variant']:<14}{r['size_mb']:>9}{r['latency_ms']:>12}{r['test_acc']:>10}")


if __name__ == "__main__":
    main()
