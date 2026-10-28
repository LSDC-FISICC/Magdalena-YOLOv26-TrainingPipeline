"""YOLO26-cls training with the rectangular 320x96 protocol of ultralytics_yolov26_train_v2.py."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import yaml
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

from . import IMG_H, IMG_W, REPO_ROOT
from .models import default_device

DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"


class MotionBlur:
    """Vertical motion blur (the direction of travel in the crop) of random length in pixels."""

    def __init__(self, max_px=24):
        self.max_px = max_px

    def __call__(self, im):
        n = int(torch.randint(2, self.max_px + 1, ()))
        k = np.zeros((n, n), np.float32)
        k[:, n // 2] = 1.0 / n
        return Image.fromarray(cv2.filter2D(np.asarray(im), -1, k, borderType=cv2.BORDER_REFLECT))


class ChannelGain:
    """Independent gain per RGB channel: white balance of daylight vs. the implement's lamps."""

    def __init__(self, delta=0.15):
        self.delta = delta

    def __call__(self, im):
        g = 1 + (torch.rand(3) * 2 - 1) * self.delta
        return Image.fromarray(np.clip(np.asarray(im, np.float32) * g.numpy(), 0, 255).astype(np.uint8))


def train_transforms():
    """Online augmentation for robustness to motion blur and illumination (night, low sun)."""
    return T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.RandomApply([MotionBlur(24)], p=0.5),
        T.RandomApply([ChannelGain(0.12)], p=0.5),
        T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.02),
        T.ToTensor(),
    ])


class RectDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix=""):
        super().__init__(root, args, augment, prefix)
        # No normalisation: YOLO-cls uses mean 0 / std 1
        self.torch_transforms = (train_transforms() if augment
                                 else T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor()]))


class RectTrainer(ClassificationTrainer):
    AUG = False  # online blur/colour augmentation of the training split

    def build_dataset(self, img_path, mode="train", batch=None):
        return RectDataset(root=img_path, args=self.args, augment=self.AUG and mode == "train", prefix=mode)


class AugRectTrainer(RectTrainer):
    AUG = True


def load_cfg(path=None, **overrides):
    cfg = yaml.safe_load(Path(path or DEFAULT_CFG).read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def train_yolo(data, project, name, model="yolo26n-cls.pt", cfg=None, seed=0, device=None, aug=False, **overrides):
    """Train and return the path of best.pt. ``cfg`` holds the shared hyper-parameters;
    ``aug`` adds online motion-blur and colour augmentation to the training split."""
    cfg = dict(cfg or load_cfg())
    cfg.update(overrides)
    weights = Path(model)
    if not weights.exists() and (REPO_ROOT / model).exists():
        weights = REPO_ROOT / model
    yolo = YOLO(str(weights))
    yolo.train(
        # absolute: Ultralytics prefixes relative projects with runs/classify/
        data=str(data), project=str(Path(project).resolve()), name=name, exist_ok=True,
        imgsz=[IMG_H, IMG_W], trainer=AugRectTrainer if aug else RectTrainer, seed=seed, deterministic=True,
        device=default_device() if device is None else device,
        # Ultralytics' own augmentations stay off: Roboflow already augmented the training
        # split, and train_transforms() covers blur and colour when aug=True
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.0, flipud=0.0, degrees=0.0,
        shear=0.0, scale=0.0, mosaic=0.0, erasing=0.0,
        save=True, plots=True, **cfg,
    )
    return Path(yolo.trainer.save_dir) / "weights" / "best.pt"
