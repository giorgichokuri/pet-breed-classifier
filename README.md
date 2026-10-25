# Pet Breed Classifier

End-to-end computer vision project: a model that recognizes **37 cat and dog breeds** from a photo,
from data preparation and experiments to an optimized INT8 model served by a FastAPI web app.

Built for the Upgaming Talent Growth Program (Machine Learning Engineer, Computer Vision).

> **Results:** see [`experiments/RESULTS.md`](experiments/RESULTS.md) (generated automatically after training)
> and the technical report in [`report/REPORT.md`](report/REPORT.md).

## What it does

1. **Data**: Oxford-IIIT Pet (7,349 images, 37 breeds). The official `trainval` split is divided
   into train (85%) and validation (15%), stratified by breed; the official `test` split (3,669 images)
   is used only for the final evaluation.
2. **Models**: a small CNN trained from scratch (baseline) vs. ImageNet-pretrained ResNet-18,
   EfficientNet-B0 and MobileNetV3 (transfer learning).
3. **Experiments**: from scratch vs. pretrained, linear probe vs. full fine-tuning, with vs. without
   augmentation, three architectures compared on accuracy, size and speed.
4. **Evaluation**: accuracy, top-5 accuracy, macro F1, cat/dog accuracy, per-breed report,
   confusion matrix, most-confused breed pairs, the most confident mistakes, and Grad-CAM heatmaps.
5. **Optimization**: ONNX export and static INT8 quantization; size, CPU latency and accuracy compared.
6. **Serving**: a FastAPI endpoint and a simple web page; inference uses ONNX Runtime only (no PyTorch).

## Quick start

```bash
git clone https://github.com/giorgichokuri/pet-breed-classifier.git
cd pet-breed-classifier
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                              # fast checks, no dataset needed
```

### Run the web app (uses the trained model in `models/`)

```bash
uvicorn app.main:app --reload
# open http://localhost:8000 and upload a photo
curl -F "file=@my_cat.jpg" "http://localhost:8000/predict?top_k=3"
```

### Predict from the command line

```bash
python -m src.predict my_dog.jpg --model models/model_int8.onnx
```

### Train

Training needs a GPU. The easiest free option is Google Colab or Kaggle with
[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb). To run everything:

```bash
bash scripts/run_all.sh        # 6 experiments + optimization + Grad-CAM + results table (~1 h on a T4)
```

Or one configuration at a time:

```bash
python -m src.train --model resnet18 --epochs 15 --run my_run          # fine-tune
python -m src.train --model resnet18 --freeze --epochs 10 --run probe  # train only the head
python -m src.evaluate --ckpt experiments/my_run/best.pt
python -m src.gradcam  --ckpt experiments/my_run/best.pt
python -m src.optimize --ckpt experiments/my_run/best.pt               # writes models/*.onnx
python scripts/summarize.py                                            # experiments/RESULTS.md
```

Quick smoke test without downloading anything: add `--fake --no-pretrained` to `src.train`.

## Experiments

| Run | What it tests |
|---|---|
| `e1_simple_cnn_scratch` | Baseline: a 0.4M-parameter CNN trained from scratch |
| `e2_resnet18_linear_probe` | Pretrained features only: backbone frozen, only the new head is trained |
| `e3_resnet18_finetune` | Full fine-tuning (head lr 1e-3, backbone lr 1e-4) |
| `e4_resnet18_finetune_noaug` | Same as e3 without data augmentation (ablation) |
| `e5_efficientnet_b0_finetune` | A more efficient architecture |
| `e6_mobilenet_v3_finetune` | A mobile architecture: smaller and faster |

The best run is selected by **validation** accuracy, then exported and quantized.

## Project structure

```
src/
  config.py      shared constants (image size, normalization, cat breeds)
  data.py        dataset download, stratified split, augmentation
  models.py      SimpleCNN + pretrained backbones with a new 37-class head
  train.py       training loop (AdamW, cosine schedule, label smoothing, AMP, early stopping)
  evaluate.py    test metrics, confusion matrix, per-class report, error analysis
  gradcam.py     Grad-CAM explanations
  optimize.py    ONNX export, INT8 quantization, size/latency/accuracy benchmark
  inference.py   ONNX Runtime predictor with numpy preprocessing (used by the API)
  predict.py     command-line prediction
app/
  main.py        FastAPI: GET / (web page), GET /health, POST /predict
  static/        web page
scripts/         run_all.sh, pick_best.py, summarize.py
notebooks/       train_colab.ipynb (free GPU)
experiments/     one folder per run: config, history, metrics, plots
models/          model.onnx (FP32), model_int8.onnx (INT8), classes.json
report/          technical report
tests/           unit and API tests
```

## Design decisions

- **Transfer learning** instead of training from scratch: 3,100 training images are too few for
  a deep network to learn visual features from zero; ImageNet features already know fur, ears and faces.
- **Two learning rates**: the new head learns 10x faster than the pretrained layers, which are only
  adjusted gently so their useful features are not destroyed.
- **Validation split for model selection**: early stopping and choosing the best run never look at
  the test set, so the test score is an honest estimate.
- **Static INT8 quantization with calibration on validation images**: about 4x smaller and faster on CPU;
  the accuracy change is measured, not assumed.
- **ONNX Runtime for serving**: the API does not need PyTorch, so it is lighter and starts faster.

## License and data

Code: MIT. Dataset: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), CC BY-SA 4.0
(Parkhi et al., 2012). Pretrained weights: torchvision (ImageNet).
