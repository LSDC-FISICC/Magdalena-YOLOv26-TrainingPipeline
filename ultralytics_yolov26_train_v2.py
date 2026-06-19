import pandas as pd
import torch
import torchvision.transforms as T  # only for Resize/ToTensor
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import glob
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import torchvision.transforms as T


# ─── Configuration ────────────────────────────────────────────────
DATASET_DIR  = "sugarcane_exg_hsv_lab_v3-2-100.v1i.folder/"
MODEL        = "yolo26s-cls.pt"
PROJECT      = "magdalena"
RUN_NAME     = "plant_cls_ExG_HSV_Lab_Curation_100_lr_0.0005_4500_WD_0.04"
EPOCHS       = 30
PATIENCE     = 5
BATCH        = 32
IMG_W        = 320
IMG_H        = 96
# ──────────────────────────────────────────────────────────────────

# Resize + ToTensor only: YOLO applies ImageNet normalization internally
INFER_TRANSFORM = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
])

# ── Automatic device detection ───────────────────────────────────
def get_device():       
    if torch.cuda.is_available():
        device = 0
        name   = torch.cuda.get_device_name(0)
        print(f"  CUDA GPU detected: {name}")
    elif torch.backends.mps.is_available():
        # MPS does not support BFloat16 required by YOLO26 MuSGD
        device = "cpu"
        print(f"  Apple Silicon detected but MPS incompatible with YOLO26 MuSGD")
        print(f"  Using CPU")
    else:
        device = "cpu"
        print(f"  No GPU — using CPU")
    return device

DEVICE = get_device()


# ── Batch size by device ─────────────────────────────────────────
BATCH_MAP = {
    0    :     64,      # CUDA GPU  ← ajusta según tu VRAM (32=4GB, 64=8GB, 128=12GB+)
    "mps": 16,      # Apple Silicon (MPS has more limited memory)
    "cpu":  8,      # CPU
}
BATCH = BATCH_MAP.get(DEVICE, 16)


# ── Custom dataset to preserve aspect ratio 320x96 ───────────────
class RectDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix=""):
        super().__init__(root, args, augment, prefix)
        # No T.Normalize: YOLOv26 applies normalization internally
        self.torch_transforms = T.Compose([
            T.Resize((IMG_H, IMG_W)),
            T.ToTensor(),
        ])


class RectTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return RectDataset(
            root    = img_path,
            args    = self.args,
            augment = False,
            prefix  = mode,
        )


# ── 1. TRAINING ───────────────────────────────────────────────────
print("=" * 60)
print("  YOLO26s — Magdalena Plant Classification")
print(f"  Train: 8,397 | Val: 799 | Test: 400")
print(f"  Image size: {IMG_W}x{IMG_H}")
print(f"  Device: {DEVICE} | Batch: {BATCH}")
print("=" * 60)
print("\n[1/4] TRAINING")

model = YOLO(MODEL)

model.train(
    data         = DATASET_DIR,
    epochs       = EPOCHS,
    patience     = PATIENCE,
    imgsz        = [IMG_H,IMG_W],
    batch        = BATCH,
    optimizer    = "MuSGD",
    device       = DEVICE,
    project      = PROJECT,
    name         = RUN_NAME,
    trainer      = RectTrainer,

    # No augmentation (already done in Roboflow)
    hsv_h        = 0.0,
    hsv_s        = 0.0,
    hsv_v        = 0.0,
    fliplr       = 0.0,
    flipud       = 0.0,
    degrees      = 0.0,
    shear        = 0.0,
    scale        = 0.0,
    mosaic       = 0.0,
    erasing      = 0.0,

    # Regularizationss
    lr0          = 0.0008, #0.001
    weight_decay = 0.01, #0.001
    dropout      = 0.5, #0.3
    workers      = 4,      # CPU threads for data loading (GPU training)
    save         = True,
    plots        = True,
    #----------------------
    # Other parameters 
    crop_fraction=1.0, # to be sure that the images are not modified
    label_smoothing=0.1,
    cos_lr=True, # to learn deeper features

)

best_weights = Path("runs") / "classify" / PROJECT / RUN_NAME / "weights" / "best.pt"
print(f"\n  Best model: {best_weights}")

# ── 2. VALIDATION ─────────────────────────────────────────────────
print("\n[2/4] VALIDATION")

model_best = YOLO(str(best_weights))
val_dir = Path(DATASET_DIR) / "valid"

# downloading of validation images
val_images = []
for class_dir in val_dir.iterdir():
    if class_dir.is_dir():
        for img_file in class_dir.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                val_images.append({"path": img_file, "true_label": class_dir.name})

print(f"Running custom validation on {len(val_images)} images...")

y_true_val = []
y_pred_val = []

for sample in val_images:
    img_pil = Image.open(sample["path"]).convert("RGB")
    tensor  = INFER_TRANSFORM(img_pil).unsqueeze(0) 

    result  = model_best.predict(source=tensor, device=DEVICE, verbose=False)[0]
    pred_label = model_best.names[result.probs.top1]
    
    y_true_val.append(sample["true_label"])
    y_pred_val.append(pred_label)

# accuracy determination
correct_val = sum(1 for t, p in zip(y_true_val, y_pred_val) if t == p)
top1_val = correct_val / len(y_true_val)
print(f"  Top-1 Accuracy (val): {top1_val*100:.2f}%")

# confusion matrix generation
classes = sorted(list(set(y_true_val)))
cm_val = confusion_matrix(y_true_val, y_pred_val, labels=classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - Validation (320x96)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
val_output_dir = Path("runs/classify") / PROJECT / (RUN_NAME + "_val")
val_output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(val_output_dir / "confusion_matrix.png", dpi=300)
plt.close()
print(f"  Confusion matrix saved to: {val_output_dir / 'confusion_matrix.png'}")

#normalised confusion matrix generation
cm_val_norm = confusion_matrix(y_true_val, y_pred_val, labels=classes, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val_norm, annot=True, fmt=".1%", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - Validation (Normalisée)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(val_output_dir / "confusion_matrix_normalized.png", dpi=300)
plt.close()


# ── 3. TEST ───────────────────────────────────────────────────────
print("\n[3/4] TEST")

test_dir = Path(DATASET_DIR) / "test"

# Dowloading of test images
test_images = []
for class_dir in test_dir.iterdir():
    if class_dir.is_dir():
        for img_file in class_dir.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                test_images.append({"path": img_file, "true_label": class_dir.name})

print(f"Running custom test on {len(test_images)} images...")

y_true_test = []
y_pred_test = []

for sample in test_images:
    img_pil = Image.open(sample["path"]).convert("RGB")
    tensor  = INFER_TRANSFORM(img_pil).unsqueeze(0)

    result  = model_best.predict(source=tensor, device=DEVICE, verbose=False)[0]
    pred_label = model_best.names[result.probs.top1]
    
    y_true_test.append(sample["true_label"])
    y_pred_test.append(pred_label)

# Accuracy determination
correct_test = sum(1 for t, p in zip(y_true_test, y_pred_test) if t == p)
top1_test = correct_test / len(y_true_test)
print(f"  Top-1 Accuracy (test): {top1_test*100:.2f}%")

# Confusion matrix generation
cm_test = confusion_matrix(y_true_test, y_pred_test, labels=classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - Test (320x96)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
test_output_dir = Path("runs/classify") / PROJECT / (RUN_NAME + "_test")
test_output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(test_output_dir / "confusion_matrix.png", dpi=300)
plt.close()
print(f"  Confusion matrix saved to: {test_output_dir / 'confusion_matrix.png'}")

# Normalised confusion matrix generation
cm_test_norm = confusion_matrix(y_true_test, y_pred_test, labels=classes, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test_norm, annot=True, fmt=".1%", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - Validation (Normalisée)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(test_output_dir / "confusion_matrix_normalized.png", dpi=300)
plt.close()
