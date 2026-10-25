# Pet Breed Classifier

End-to-end computer vision project: a model that recognizes **37 cat and dog breeds** from a photo,
from data preparation and experiments to an optimized ONNX model served by a FastAPI web app.

Built for the Upgaming Talent Growth Program (Machine Learning Engineer, Computer Vision).

## Results

Test set: 3,669 images, 37 breeds. The best run was chosen by **validation** accuracy.

| Run | Params | Val acc | Test acc | Top-5 | Macro F1 | Cat/dog acc |
|---|---:|---:|---:|---:|---:|---:|
| e1 SimpleCNN from scratch | 0.4M | 42.9% | 30.5% | 67.9% | 0.296 | 78.6% |
| e2 ResNet-18, linear probe | 11.2M (0.02M trained) | 90.8% | 88.8% | 99.0% | 0.886 | 99.0% |
| e3 ResNet-18, fine-tune | 11.2M | 93.1% | 88.1% | 98.4% | 0.878 | 99.1% |
| e4 ResNet-18, fine-tune, no augmentation | 11.2M | 92.8% | 89.4% | 98.2% | 0.891 | 99.2% |
| **e5 EfficientNet-B0, fine-tune** | **4.05M** | **94.0%** | **91.3%** | **99.3%** | **0.912** | **99.6%** |
| e6 MobileNetV3-Large, fine-tune | 4.25M | 93.7% | 89.6% | 99.1% | 0.895 | 99.3% |

Deployment (EfficientNet-B0, CPU, 1 thread, batch 1):

| Variant | Size | Latency | Test acc |
|---|---:|---:|---:|
| PyTorch FP32 | 16.5 MB | 38.6 ms | 91.3% |
| **ONNX FP32 (served)** | **16.2 MB** | **15.8 ms** | **91.3%** |
| ONNX INT8, best of 4 recipes | 4.8 MB | 29.4 ms | 40.3% |

Static INT8 quantization breaks EfficientNet-B0 (and was not faster on this CPU), while the same
pipeline keeps ResNet-18 within about 1 point, so the API serves the FP32 ONNX model.
Full tables: [`experiments/RESULTS.md`](experiments/RESULTS.md); analysis: [`report/REPORT.md`](report/REPORT.md).

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
5. **Optimization**: ONNX export (2.4x faster than PyTorch on CPU) and a static INT8 quantization study;
   size, CPU latency and accuracy compared.
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
python -m src.predict my_dog.jpg --model models/model.onnx
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

The best run is selected by **validation** accuracy, then exported to ONNX and quantized.
`scripts/quant_diag.py` checks whether an INT8 accuracy drop comes from the pipeline or the architecture.

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
scripts/         run_all.sh, pick_best.py, summarize.py, quant_diag.py
notebooks/       train_colab.ipynb (free GPU)
experiments/     one folder per run: config, history, metrics, plots
models/          model.onnx (FP32, served), model_int8.onnx (INT8, not used: see results), classes.json
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
- **INT8 is measured, not assumed**: several static quantization recipes are calibrated on validation
  images and chosen by validation accuracy. For EfficientNet-B0 every recipe lost more than half of the
  accuracy, so the FP32 ONNX model is served; ResNet-18 quantizes fine with the same code.
- **ONNX Runtime for serving**: the API does not need PyTorch, so it is lighter and starts faster.

## License and data

Code: MIT. Dataset: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), CC BY-SA 4.0
(Parkhi et al., 2012). Pretrained weights: torchvision (ImageNet).
