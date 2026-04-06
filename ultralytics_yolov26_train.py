import pandas as pd
import torch
import torchvision.transforms as T  # only for Resize/ToTensor
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Configuration ────────────────────────────────────────────────
DATASET_DIR  = "dataset/"
MODEL        = "yolo26s-cls.pt"
PROJECT      = "magdalena"
RUN_NAME     = "plant_cls_v4"
EPOCHS       = 15
PATIENCE     = 5
BATCH        = 32
IMG_W        = 320
IMG_H        = 96
# ──────────────────────────────────────────────────────────────────


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
    0:     64,      # CUDA GPU  ← ajusta según tu VRAM (32=4GB, 64=8GB, 128=12GB+)
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
    imgsz        = IMG_W,
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

    # Regularization
    lr0          = 0.001,
    weight_decay = 0.001,
    dropout      = 0.3,

    workers      = 8,      # CPU threads for data loading (GPU training)
    save         = True,
    plots        = True,
)

best_weights = Path("runs") / "classify" / PROJECT / RUN_NAME / "weights" / "best.pt"
print(f"\n  Best model: {best_weights}")


# ── 2. VALIDATION ─────────────────────────────────────────────────
print("\n[2/4] VALIDATION")

model_best = YOLO(str(best_weights))

val_results = model_best.val(
    data    = DATASET_DIR,
    imgsz   = IMG_W,
    batch   = BATCH,
    device  = DEVICE,
    project = PROJECT,
    name    = RUN_NAME + "_val",
    split   = "val",
)

top1_val = val_results.top1
top5_val = val_results.top5
print(f"  Top-1 Accuracy (val): {top1_val*100:.2f}%")
print(f"  Top-5 Accuracy (val): {top5_val*100:.2f}%")


# ── 3. TEST ───────────────────────────────────────────────────────
print("\n[3/4] TEST")

test_results = model_best.val(
    data    = DATASET_DIR,
    imgsz   = IMG_W,
    batch   = BATCH,
    device  = DEVICE,
    project = PROJECT,
    name    = RUN_NAME + "_test",
    split   = "test",
)

top1_test = test_results.top1
top5_test = test_results.top5
print(f"  Top-1 Accuracy (test): {top1_test*100:.2f}%")
print(f"  Top-5 Accuracy (test): {top5_test*100:.2f}%")


# ── 4. PLOTS ──────────────────────────────────────────────────────
print("\n[4/4] PLOTTING")

results_csv = Path("runs") / "classify" / PROJECT / RUN_NAME / "results.csv"
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()

fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    f"YOLO26s — Magdalena Plant Classification\n"
    f"Train: 8,397 | Val: 799 | Test: 400 | {IMG_W}x{IMG_H}px | device: {DEVICE}",
    fontsize=13, fontweight="bold"
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(df["epoch"], df["train/loss"], label="Train", color="royalblue", linewidth=2)
ax1.plot(df["epoch"], df["val/loss"],   label="Val",   color="tomato",    linewidth=2, linestyle="--")
ax1.set_title("Loss — Train vs Val")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2])
if "lr/pg0" in df.columns:
    ax2.plot(df["epoch"], df["lr/pg0"], color="slateblue", linewidth=2)
ax2.set_title("Learning Rate")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("LR")
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(df["epoch"], df["metrics/accuracy_top1"], color="seagreen", linewidth=2)
ax3.axhline(y=top1_val,  color="seagreen",  linestyle="--", alpha=0.7, label=f"Val  {top1_val*100:.1f}%")
ax3.axhline(y=top1_test, color="darkorange", linestyle=":",  alpha=0.7, label=f"Test {top1_test*100:.1f}%")
ax3.set_title("Top-1 Accuracy")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Accuracy")
ax3.set_ylim(0, 1)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(df["epoch"], df["metrics/accuracy_top5"], color="darkorange", linewidth=2)
ax4.axhline(y=top5_val,  color="seagreen",  linestyle="--", alpha=0.7, label=f"Val  {top5_val*100:.1f}%")
ax4.axhline(y=top5_test, color="darkorange", linestyle=":",  alpha=0.7, label=f"Test {top5_test*100:.1f}%")
ax4.set_title("Top-5 Accuracy")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Accuracy")
ax4.set_ylim(0, 1)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
tabla = [
    ["Split", "Top-1",              "Top-5"],
    ["Val",   f"{top1_val*100:.1f}%",  f"{top5_val*100:.1f}%"],
    ["Test",  f"{top1_test*100:.1f}%", f"{top5_test*100:.1f}%"],
]
t = ax5.table(cellText=tabla[1:], colLabels=tabla[0],
              cellLoc="center", loc="center")
t.auto_set_font_size(False)
t.set_fontsize(10)
t.scale(1.2, 2.0)
ax5.set_title("Final Results", fontweight="bold")

save_dir  = Path("runs") / "classify" / PROJECT / RUN_NAME / "saved_plots"
save_dir.mkdir(parents=True, exist_ok=True)
plot_path = save_dir / "training_metrics.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Model:         {MODEL}")
print(f"  Device:        {DEVICE}")
print(f"  Batch size:    {BATCH}")
print(f"  Image:         {IMG_W}x{IMG_H}")
print(f"  Epochs:        {EPOCHS}")
print(f"\n  {'Split':<8} {'Top-1':>8} {'Top-5':>8}")
print(f"  {'-'*28}")
print(f"  {'Val':<8} {top1_val*100:>7.2f}% {top5_val*100:>7.2f}%")
print(f"  {'Test':<8} {top1_test*100:>7.2f}% {top5_test*100:>7.2f}%")
print(f"\n  Best model:    {best_weights}")
print(f"  Plot:          {plot_path}")
print("=" * 60)

# ── Copy images generated by YOLO26 to saved_plots ───────────────
import shutil

runs_dir = Path("runs") / "classify" / PROJECT / RUN_NAME

img_extensions = ["*.png", "*.jpg", "*.jpeg"]

copied = 0
for ext in img_extensions:
    for img_file in runs_dir.rglob(ext):
        if save_dir in img_file.parents:
            continue
        dst = save_dir / img_file.name
        if dst.exists():
            dst = save_dir / f"{img_file.stem}_{img_file.parent.name}{img_file.suffix}"
        shutil.copy(img_file, dst)
        copied += 1
        print(f"  Saved: {dst.name}")

print(f"\n  Total images saved: {copied}")
print(f"  Folder: {save_dir}")
