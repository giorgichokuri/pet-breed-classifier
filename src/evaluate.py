"""Evaluate a checkpoint on the test set and produce an error analysis.

Outputs (in the run folder): metrics.json values, per_class.csv, confusion_matrix.png,
most_confused.csv, misclassified.png.

    python -m src.evaluate --ckpt experiments/resnet18_finetune/best.pt
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import IMAGENET_MEAN, IMAGENET_STD, species_of
from .data import load_datasets, make_loaders
from .utils import get_device, load_checkpoint, write_json


@torch.no_grad()
def predict_all(model, loader, device):
    """Return true labels, predicted labels and class probabilities for a whole loader."""
    ys, probs = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1).cpu()
        probs.append(p)
        ys.append(y)
    probs = torch.cat(probs).numpy()
    y_true = torch.cat(ys).numpy()
    return y_true, probs.argmax(1), probs


def top_k_accuracy(y_true, probs, k=5):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, topk)]))


def plot_confusion(cm, classes, path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def denormalize(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_misclassified(dataset, y_true, y_pred, probs, classes, path, n=16):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    # show the most confident mistakes: the most interesting ones to analyze
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:n]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        img, _ = dataset[i]
        ax.imshow(denormalize(img))
        ax.set_title(f"true: {classes[y_true[i]]}\npred: {classes[y_pred[i]]} ({probs[i].max():.0%})", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate_checkpoint(ckpt_path, out_dir, data_dir="data", fake=False, batch_size=64, num_workers=2):
    device = get_device()
    model, model_name, classes = load_checkpoint(ckpt_path, device)
    _, _, test_ds, _ = load_datasets(data_dir, "none", fake=fake)
    _, _, test_dl = make_loaders(test_ds, test_ds, test_ds, batch_size, num_workers)

    y_true, y_pred, probs = predict_all(model, test_dl, device)
    out_dir = Path(out_dir)
    labels = list(range(len(classes)))

    report = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.round(4).to_csv(out_dir / "per_class.csv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion(cm, classes, out_dir / "confusion_matrix.png")

    # which pairs of breeds get mixed up most often
    pairs = Counter((classes[t], classes[p]) for t, p in zip(y_true, y_pred) if t != p)
    pd.DataFrame([{"true": a, "predicted": b, "count": c} for (a, b), c in pairs.most_common(15)]) \
        .to_csv(out_dir / "most_confused.csv", index=False)

    plot_misclassified(test_ds, y_true, y_pred, probs, classes, out_dir / "misclassified.png")

    species_true = np.array([species_of(classes[i]) for i in y_true])
    species_pred = np.array([species_of(classes[i]) for i in y_pred])
    metrics = {
        "test_acc": round(accuracy_score(y_true, y_pred), 4),
        "test_top5": round(top_k_accuracy(y_true, probs, 5), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "species_acc": round(float(np.mean(species_true == species_pred)), 4),
        "n_test": int(len(y_true)),
    }
    write_json(out_dir / "test_metrics.json", metrics)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--fake", action="store_true")
    a = p.parse_args()
    print(evaluate_checkpoint(a.ckpt, Path(a.ckpt).parent, a.data_dir, a.fake))


if __name__ == "__main__":
    main()
