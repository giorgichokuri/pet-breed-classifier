"""Command-line prediction for one image.

    python -m src.predict path/to/photo.jpg --model models/model.onnx
"""
import argparse
import json

from PIL import Image

from .inference import Predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--model", default="models/model.onnx")
    p.add_argument("--top-k", type=int, default=5)
    a = p.parse_args()
    result = Predictor(a.model).predict(Image.open(a.image), a.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
