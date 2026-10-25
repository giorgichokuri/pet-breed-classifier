"""Train one model configuration and evaluate it on the test set.

Example:
    python -m src.train --model resnet18 --epochs 15 --run resnet18_finetune
    python -m src.train --model resnet18 --freeze --epochs 10 --run resnet18_linear_probe
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from .data import load_datasets, make_loaders
from .evaluate import evaluate_checkpoint
from .models import MODEL_NAMES, build_model, count_parameters, head_parameters
from .utils import get_device, save_checkpoint, set_seed, write_json
from .config import SEED


def parse_args():
    p = argparse.ArgumentParser(description="Train a pet breed classifier")
    p.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    p.add_argument("--run", default=None, help="run name; outputs go to experiments/<run>")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate of the new head")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="pretrained layers learn this many times slower than the head")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--aug", choices=["none", "basic", "strong"], default="basic")
    p.add_argument("--freeze", action="store_true", help="train only the head (linear probe)")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--fake", action="store_true", help="random images, for a quick smoke test")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over the data. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    use_amp = device.type == "cuda"
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()      # backpropagation
                scaler.step(optimizer)             # update the weights
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    run = args.run or f"{args.model}{'_frozen' if args.freeze else ''}_{args.aug}"
    out = Path("experiments") / run
    out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, classes = load_datasets(args.data_dir, args.aug, fake=args.fake)
    train_dl, val_dl, _ = make_loaders(train_ds, val_ds, test_ds, args.batch_size, args.num_workers)
    print(f"device={device} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} | classes={len(classes)}")

    pretrained = not args.no_pretrained and args.model != "simple_cnn"
    model = build_model(args.model, len(classes), pretrained=pretrained, freeze_backbone=args.freeze).to(device)

    # Two learning rates: the new head learns fast, pretrained layers are only nudged.
    head = head_parameters(model, args.model)
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": args.lr}]
    if backbone and pretrained:
        groups.append({"params": backbone, "lr": args.lr * args.backbone_lr_mult})
    elif backbone:  # model trained from scratch: everything uses the same learning rate
        groups[0]["params"] = list(model.parameters())

    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    print(f"model={args.model} pretrained={pretrained} frozen={args.freeze} "
          f"params={count_parameters(model):,} trainable={count_parameters(model, True):,}")

    best_acc, best_epoch, bad_epochs = -1.0, 0, 0
    history = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        row = dict(epoch=epoch, train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 4),
                   val_loss=round(va_loss, 4), val_acc=round(va_acc, 4), seconds=round(time.time() - t0, 1))
        history.append(row)
        print(row)

        if va_acc > best_acc:   # keep the epoch with the best validation accuracy
            best_acc, best_epoch, bad_epochs = va_acc, epoch, 0
            save_checkpoint(out / "best.pt", model, args.model, classes)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping: no improvement for {args.patience} epochs")
                break

    with open(out / "history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    train_minutes = (time.time() - t_start) / 60
    config = {k: v for k, v in vars(args).items()}
    config.update(run=run, pretrained=pretrained, device=str(device),
                  params=count_parameters(model), trainable_params=count_parameters(model, True),
                  best_epoch=best_epoch, best_val_acc=round(best_acc, 4),
                  epochs_run=len(history), train_minutes=round(train_minutes, 2))
    write_json(out / "config.json", config)

    print(f"best val acc {best_acc:.4f} at epoch {best_epoch}; evaluating on the test set...")
    metrics = evaluate_checkpoint(out / "best.pt", out, data_dir=args.data_dir, fake=args.fake,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
    metrics.update({k: config[k] for k in ("run", "model", "pretrained", "freeze", "aug", "params",
                                            "trainable_params", "best_epoch", "best_val_acc",
                                            "epochs_run", "train_minutes")})
    write_json(out / "metrics.json", metrics)
    print({k: metrics[k] for k in ("test_acc", "test_top5", "macro_f1")})


if __name__ == "__main__":
    main()
