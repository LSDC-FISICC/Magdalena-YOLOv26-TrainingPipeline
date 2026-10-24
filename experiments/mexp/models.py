"""Uniform batched inference for YOLO (.pt/.engine/.onnx) and torchvision classifiers.

All backends receive the same input as the deployed C++ node: the frame is
stretched to 320x96 (as Roboflow did when building the dataset), scaled to
[0, 1] and fed as RGB NCHW. YOLO-cls uses mean 0 / std 1, torchvision models
add ImageNet normalisation.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import CLASSES, IMG_H, IMG_W

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_tensor(path):
    img = Image.open(path).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def default_device():
    return 0 if torch.cuda.is_available() else "cpu"


class Predictor:
    """``predict(paths) -> (labels, p_plant)``; picks the backend from the file suffix."""

    def __init__(self, weights, device=None, batch=5, half=None):
        self.weights = str(weights)
        self.device = default_device() if device is None else device
        self.batch = batch
        suffix = Path(self.weights).suffix
        if suffix == ".pth":
            from .tv_models import load_checkpoint
            self.kind = "torchvision"
            self.model = load_checkpoint(self.weights).eval()
            self.torch_device = torch.device("cuda" if self.device != "cpu" else "cpu")
            self.model.to(self.torch_device)
        else:
            from ultralytics import YOLO
            self.kind = "yolo"
            self.model = YOLO(self.weights, task="classify")
            names = getattr(self.model, "names", None) or {0: CLASSES[0], 1: CLASSES[1]}
            self.plant_idx = [k for k, v in names.items() if v == "plant"][0]
            self.half = suffix == ".engine" if half is None else half

    @torch.no_grad()
    def predict_tensor(self, x):
        """x: float tensor (B,3,96,320) in [0,1]. Returns p(plant) per image as numpy."""
        if self.kind == "torchvision":
            x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(self.torch_device)
            return torch.softmax(self.model(x), 1)[:, 1].float().cpu().numpy()
        res = self.model.predict(source=x, imgsz=(IMG_H, IMG_W), device=self.device,
                                 half=self.half, verbose=False)
        return np.array([float(r.probs.data[self.plant_idx]) for r in res])

    def predict(self, paths, threshold=0.5):
        probs = []
        for i in range(0, len(paths), self.batch):
            x = torch.stack([load_tensor(p) for p in paths[i:i + self.batch]])
            probs.extend(self.predict_tensor(x).tolist())
        labels = ["plant" if p >= threshold else "no_plant" for p in probs]
        return labels, probs
