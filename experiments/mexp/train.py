"""YOLO26-cls training with the rectangular 320x96 protocol of ultralytics_yolov26_train_v2.py."""

from pathlib import Path

import torchvision.transforms as T
import yaml
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

from . import IMG_H, IMG_W, REPO_ROOT
from .models import default_device

DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"


class RectDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix=""):
        super().__init__(root, args, augment, prefix)
        # No normalisation: YOLO-cls uses mean 0 / std 1
        self.torch_transforms = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor()])


class RectTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return RectDataset(root=img_path, args=self.args, augment=False, prefix=mode)


def load_cfg(path=None, **overrides):
    cfg = yaml.safe_load(Path(path or DEFAULT_CFG).read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def train_yolo(data, project, name, model="yolo26n-cls.pt", cfg=None, seed=0, device=None, **overrides):
    """Train and return the path of best.pt. ``cfg`` holds the shared hyper-parameters."""
    cfg = dict(cfg or load_cfg())
    cfg.update(overrides)
    weights = Path(model)
    if not weights.exists() and (REPO_ROOT / model).exists():
        weights = REPO_ROOT / model
    yolo = YOLO(str(weights))
    yolo.train(
        # absolute: Ultralytics prefixes relative projects with runs/classify/
        data=str(data), project=str(Path(project).resolve()), name=name, exist_ok=True,
        imgsz=[IMG_H, IMG_W], trainer=RectTrainer, seed=seed, deterministic=True,
        device=default_device() if device is None else device,
        # Roboflow already augmented the training split
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.0, flipud=0.0, degrees=0.0,
        shear=0.0, scale=0.0, mosaic=0.0, erasing=0.0,
        save=True, plots=True, **cfg,
    )
    return Path(yolo.trainer.save_dir) / "weights" / "best.pt"
