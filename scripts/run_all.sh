#!/usr/bin/env bash
# Runs every experiment from the report, then optimizes and explains the best model.
# On a free Colab/Kaggle T4 GPU the whole script takes roughly 1 hour.
set -euo pipefail

python -m src.train --model simple_cnn          --epochs 30 --lr 3e-3 --patience 8 --run e1_simple_cnn_scratch
python -m src.train --model resnet18 --freeze   --epochs 10                         --run e2_resnet18_linear_probe
python -m src.train --model resnet18            --epochs 15                         --run e3_resnet18_finetune
python -m src.train --model resnet18            --epochs 15 --aug none              --run e4_resnet18_finetune_noaug
python -m src.train --model efficientnet_b0     --epochs 15                         --run e5_efficientnet_b0_finetune
python -m src.train --model mobilenet_v3_large  --epochs 15                         --run e6_mobilenet_v3_finetune

# The best model is chosen by VALIDATION accuracy (never by test accuracy).
BEST=$(python scripts/pick_best.py)
echo "Best run: $BEST"
python -m src.optimize --ckpt "experiments/$BEST/best.pt"
python -m src.gradcam  --ckpt "experiments/$BEST/best.pt" --n 8
python scripts/summarize.py
