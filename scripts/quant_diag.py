"""Diagnose INT8 accuracy loss: is it our pipeline or the architecture?

    python scripts/quant_diag.py

1. ResNet-18 (ReLU, no squeeze-and-excitation) with the same pipeline and min/max calibration:
   if it keeps its accuracy, the pipeline is correct and the problem is EfficientNet-specific.
2. EfficientNet-B0 with histogram calibration (Percentile, Entropy), which clips rare outlier
   activations instead of stretching the 8-bit range to cover them. Histogram calibration stores
   intermediate outputs in memory, so it uses fewer calibration images.
All numbers are validation accuracy; the test set is not used here.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static  # noqa: E402
from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: E402

from src.data import load_datasets  # noqa: E402
from src.optimize import ImageReader, accuracy, export_onnx, ort_fn, session  # noqa: E402
from src.utils import load_checkpoint  # noqa: E402

OUT = Path("experiments/quant_diag")
CASES = [
    ("e3_resnet18_finetune", CalibrationMethod.MinMax, 200),
    ("e5_efficientnet_b0_finetune", CalibrationMethod.Percentile, 32),
    ("e5_efficientnet_b0_finetune", CalibrationMethod.Entropy, 32),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, val_ds, _, _ = load_datasets("data", "none", fake="--fake" in sys.argv)
    rows = []
    for run, method, n in CASES:
        model, _, _ = load_checkpoint(f"experiments/{run}/best.pt")
        fp32, prep = OUT / f"{run}.onnx", OUT / f"{run}_prep.onnx"
        if not prep.exists():
            export_onnx(model, fp32)
            quant_pre_process(str(fp32), str(prep))
        int8 = OUT / f"{run}_{method.name}_{n}.onnx"
        row = {"run": run, "calibration": method.name, "calib_images": n,
               "fp32_val_acc": accuracy(ort_fn(session(prep)), val_ds)}
        try:
            quantize_static(str(prep), str(int8), ImageReader(val_ds, n), quant_format=QuantFormat.QDQ,
                            per_channel=True, calibrate_method=method,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
            row["int8_val_acc"] = accuracy(ort_fn(session(int8)), val_ds)
        except Exception as e:  # e.g. a histogram calibrator failing on a nearly constant tensor
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print(row, flush=True)
    for f in OUT.glob("*.onnx"):
        f.unlink()
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
