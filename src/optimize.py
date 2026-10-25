"""Optimization: export to ONNX, quantize to INT8 and compare size / speed / accuracy.

    python -m src.optimize --ckpt experiments/e5_efficientnet_b0_finetune/best.pt

Produces in models/: model.onnx (FP32), model_int8.onnx (INT8), classes.json,
and experiments/<run>/optimization.json with the comparison tables.

Why: FP32 stores every weight in 4 bytes; INT8 stores it in 1 byte. The model gets ~4x
smaller and often faster on CPU, at the cost of an accuracy drop that we measure.

Static INT8 quantization is sensitive to the architecture. Quantizing every operator with
min/max ranges works for plain CNNs such as ResNet, but EfficientNet's SiLU activations and
squeeze-and-excitation blocks (Sigmoid, Mul) have outlier values, and accuracy can collapse.
So we try several quantization recipes, choose the one with the best VALIDATION accuracy
(the test set is never used for choices) and report its test accuracy.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from .config import IMG_SIZE
from .data import load_datasets
from .utils import load_checkpoint, write_json

# name -> (operator types to quantize (None = all supported), calibration method)
RECIPES = {
    "all_ops_minmax": (None, CalibrationMethod.MinMax),
    "conv_gemm_minmax": (["Conv", "Gemm", "MatMul"], CalibrationMethod.MinMax),
    "conv_gemm_percentile": (["Conv", "Gemm", "MatMul"], CalibrationMethod.Percentile),
    "conv_gemm_entropy": (["Conv", "Gemm", "MatMul"], CalibrationMethod.Entropy),
}


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
        step = max(1, len(dataset) // n)      # spread over the whole set, so every breed is seen
        self.items = iter([{"image": dataset[i][0].unsqueeze(0).numpy()} for i in range(0, len(dataset), step)][:n])

    def get_next(self):
        return next(self.items, None)


def quantize(prep_path: Path, int8_path: Path, calib_ds, op_types=None, method=CalibrationMethod.MinMax):
    kwargs = dict(quant_format=QuantFormat.QDQ, per_channel=True, calibrate_method=method,
                  activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    if op_types:
        kwargs["op_types_to_quantize"] = op_types
    quantize_static(str(prep_path), str(int8_path), ImageReader(calib_ds), **kwargs)


def latency_ms(fn, x, runs=50, warmup=5):
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 2)


def accuracy(fn, dataset, limit=0):
    n = min(limit, len(dataset)) if limit else len(dataset)
    correct = 0
    for i in range(n):
        x, y = dataset[i]
        correct += int(np.argmax(fn(x.unsqueeze(0).numpy())) == y)
    return round(correct / n, 4)


def session(path: Path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def ort_fn(sess):
    return lambda x: sess.run(None, {"image": x})[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="models")
    p.add_argument("--limit", type=int, default=0, help="evaluate on the first N test images (0 = all)")
    p.add_argument("--recipes", nargs="+", default=list(RECIPES), choices=list(RECIPES))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    model, model_name, classes = load_checkpoint(a.ckpt, "cpu")
    _, val_ds, test_ds, _ = load_datasets(a.data_dir, "none", fake=a.fake)

    fp32, int8, prep = out / "model.onnx", out / "model_int8.onnx", out / "model_prep.onnx"
    export_onnx(model, fp32)
    quant_pre_process(str(fp32), str(prep))
    (out / "classes.json").write_text(json.dumps(classes))
    sess32 = session(fp32)

    @torch.no_grad()
    def torch_fn(x):
        return model(torch.from_numpy(x)).numpy()

    x1 = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    # sanity check: the ONNX model must give the same outputs as PyTorch
    max_diff = float(np.abs(torch_fn(x1) - ort_fn(sess32)(x1)).max())

    # 1) try every quantization recipe; calibrate on validation images, choose by validation accuracy
    fp32_val = accuracy(ort_fn(sess32), val_ds)
    sweep = []
    for name in a.recipes:
        op_types, method = RECIPES[name]
        cand = out / f"model_int8_{name}.onnx"
        t0 = time.time()
        quantize(prep, cand, val_ds, op_types, method)
        sess = session(cand)
        row = {"recipe": name, "val_acc": accuracy(ort_fn(sess), val_ds),
               "size_mb": round(cand.stat().st_size / 1e6, 2), "latency_ms": latency_ms(ort_fn(sess), x1),
               "quantize_s": round(time.time() - t0, 1)}
        sweep.append(row)
        print(row)
    best = max(sweep, key=lambda r: r["val_acc"])
    shutil.copy(out / f"model_int8_{best['recipe']}.onnx", int8)
    for name in a.recipes:
        (out / f"model_int8_{name}.onnx").unlink(missing_ok=True)
    prep.unlink(missing_ok=True)
    print(f"chosen recipe: {best['recipe']} (val acc {best['val_acc']} vs FP32 {fp32_val})")

    # 2) final comparison on the test set
    sess8 = session(int8)
    ckpt_mb = Path(a.ckpt).stat().st_size / 1e6
    rows = [
        {"variant": "PyTorch FP32", "size_mb": round(ckpt_mb, 2), "latency_ms": latency_ms(torch_fn, x1),
         "test_acc": accuracy(torch_fn, test_ds, a.limit)},
        {"variant": "ONNX FP32", "size_mb": round(fp32.stat().st_size / 1e6, 2),
         "latency_ms": latency_ms(ort_fn(sess32), x1), "test_acc": accuracy(ort_fn(sess32), test_ds, a.limit)},
        {"variant": f"ONNX INT8 ({best['recipe']})", "size_mb": round(int8.stat().st_size / 1e6, 2),
         "latency_ms": latency_ms(ort_fn(sess8), x1), "test_acc": accuracy(ort_fn(sess8), test_ds, a.limit)},
    ]
    result = {"model": model_name, "onnx_vs_torch_max_abs_diff": max_diff,
              "eval_images": a.limit or len(test_ds), "cpu_threads": torch.get_num_threads(),
              "fp32_val_acc": fp32_val, "chosen_recipe": best["recipe"], "quantization_sweep": sweep,
              "variants": rows}
    write_json(Path(a.ckpt).parent / "optimization.json", result)

    print(f"ONNX vs PyTorch max |diff| = {max_diff:.2e}")
    print(f"{'variant':<34}{'size MB':>9}{'latency ms':>12}{'test acc':>10}")
    for r in rows:
        print(f"{r['variant']:<34}{r['size_mb']:>9}{r['latency_ms']:>12}{r['test_acc']:>10}")


if __name__ == "__main__":
    main()
