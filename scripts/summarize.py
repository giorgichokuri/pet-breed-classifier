"""Collect all runs into experiments/RESULTS.md and plot validation curves (experiments/curves.png)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXP = Path("experiments")
rows, curves = [], {}
for f in sorted(EXP.glob("*/metrics.json")):
    run = f.parent.name
    if run.startswith("smoke"):
        continue
    m = json.loads(f.read_text())
    rows.append({
        "run": run, "model": m["model"], "pretrained": m["pretrained"], "frozen": m["freeze"], "aug": m["aug"],
        "params (M)": round(m["params"] / 1e6, 2), "trainable (M)": round(m["trainable_params"] / 1e6, 2),
        "epochs": m["epochs_run"], "best epoch": m["best_epoch"], "val acc": m["best_val_acc"],
        "test acc": m["test_acc"], "top-5": m["test_top5"], "macro F1": m["macro_f1"],
        "cat/dog acc": m["species_acc"], "train min": m["train_minutes"],
    })
    hist = f.parent / "history.csv"
    if hist.exists():
        curves[run] = pd.read_csv(hist)

if not rows:
    raise SystemExit("no finished runs")

lines = ["# Results", "", "## Experiments (test set: 3,669 images, 37 breeds)", "",
         pd.DataFrame(rows).to_markdown(index=False), ""]

for f in sorted(EXP.glob("*/optimization.json")):
    o = json.loads(f.read_text())
    lines += [f"## Optimization: {f.parent.name} ({o['model']})", "",
              f"ONNX vs PyTorch max abs difference: {o['onnx_vs_torch_max_abs_diff']:.1e}; "
              f"evaluated on {o['eval_images']} test images; CPU threads: {o['cpu_threads']}", "",
              pd.DataFrame(o["variants"]).to_markdown(index=False), ""]
    if "quantization_sweep" in o:
        lines += [f"### INT8 quantization recipes (chosen by validation accuracy: {o['chosen_recipe']}; "
                  f"FP32 validation accuracy {o['fp32_val_acc']})", "",
                  pd.DataFrame(o["quantization_sweep"]).to_markdown(index=False), ""]
    conf = f.parent / "most_confused.csv"
    if conf.exists():
        lines += ["### Most confused breed pairs (best model)", "",
                  pd.read_csv(conf).head(10).to_markdown(index=False), ""]

(EXP / "RESULTS.md").write_text("\n".join(lines))
print("\n".join(lines))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for run, h in curves.items():
    axes[0].plot(h["epoch"], h["val_acc"], marker="o", ms=3, label=run)
    axes[1].plot(h["epoch"], h["train_loss"], ls="--", alpha=0.6, label=f"{run} train")
    axes[1].plot(h["epoch"], h["val_loss"], label=f"{run} val")
axes[0].set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy")
axes[1].set(title="Loss (dashed = train)", xlabel="epoch", ylabel="loss")
axes[0].legend(fontsize=7)
axes[1].legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(EXP / "curves.png", dpi=120)
print("saved", EXP / "curves.png")
