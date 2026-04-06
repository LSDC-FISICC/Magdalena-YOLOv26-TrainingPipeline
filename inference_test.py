import random
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────
MODEL_PATH = "runs/classify/magdalena/plant_cls_v3/weights/best.pt"
TEST_DIR   = "dataset/test"
N_IMAGES   = 20
DEVICE     = 0 if torch.cuda.is_available() else "cpu"
SEED       = 42
IMG_W      = 320
IMG_H      = 96
# ──────────────────────────────────────────────────────────────────

# Resize + ToTensor only: YOLO applies ImageNet normalization internally
INFER_TRANSFORM = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
])

COLORS = {
    "plant":    "#00C853",
    "no_plant": "#FF1744",
}

# ── Load model ────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")
print(f"Classes: {model.names}")

# ── Collect images from the test set ─────────────────────────────
test_path  = Path(TEST_DIR)
all_images = []

for class_dir in test_path.iterdir():
    if class_dir.is_dir():
        for img_file in class_dir.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                all_images.append({
                    "path":       img_file,
                    "true_label": class_dir.name
                })

random.seed(SEED)
samples = random.sample(all_images, min(N_IMAGES, len(all_images)))
print(f"\nTotal images in test set: {len(all_images)}")
print(f"Selected images: {len(samples)}")

# ── Inference ─────────────────────────────────────────────────────
print("\nRunning inference...")
results_data = []

for sample in samples:
    # Preprocess same as training: 320x96 + ImageNet normalization
    img_pil = Image.open(sample["path"]).convert("RGB")
    tensor  = INFER_TRANSFORM(img_pil).unsqueeze(0)  # [1, 3, 96, 320]

    result  = model.predict(
        source  = tensor,
        device  = DEVICE,
        verbose = False
    )[0]

    probs      = result.probs
    top1_idx   = probs.top1
    top1_conf  = float(probs.top1conf)
    pred_label = model.names[top1_idx]

    results_data.append({
        "path":       sample["path"],
        "true_label": sample["true_label"],
        "pred_label": pred_label,
        "pred_idx":   top1_idx,
        "confidence": top1_conf,
        "correct":    pred_label == sample["true_label"],
    })

# ── Summary table in console ──────────────────────────────────────
print(f"\n{'#':<4} {'File':<35} {'True':<12} {'Prediction':<12} {'Idx':<6} {'Confidence':<12} {'OK'}")
print("-" * 90)

for i, r in enumerate(results_data):
    ok = "+" if r["correct"] else "x"
    print(
        f"{i+1:<4} "
        f"{r['path'].name:<35} "
        f"{r['true_label']:<12} "
        f"{r['pred_label']:<12} "
        f"{r['pred_idx']:<6} "
        f"{r['confidence']*100:.2f}%{'':6} "
        f"{ok}"
    )

n_correct = sum(1 for r in results_data if r["correct"])
print("-" * 90)
print(f"\nSample accuracy: {n_correct}/{len(results_data)} ({n_correct/len(results_data)*100:.0f}%)")

# ── Visualization ─────────────────────────────────────────────────
print("\nGenerating visualization...")

fig, axes = plt.subplots(
    2, 5,
    figsize    = (22, 10),
    facecolor  = "#0D0D0D"
)

fig.suptitle(
    "YOLO26s — Inferencia en Test Set   |   Proyecto Magdalena",
    fontsize   = 15,
    fontweight = "bold",
    color      = "white",
    y          = 1.01
)

for ax, result in zip(axes.flat, results_data):

    img    = Image.open(result["path"]).convert("RGB")
    img_np = np.array(img)

    color    = COLORS[result["pred_label"]]
    is_plant = result["pred_label"] == "plant"
    correct  = result["correct"]

    # Display image
    ax.imshow(img_np)
    ax.set_facecolor("#0D0D0D")

    # Border on spines
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(6)

    # Rectangle overlay on image
    h, w = img_np.shape[:2]
    rect = patches.Rectangle(
        (0, 0), w, h,
        linewidth = 7,
        edgecolor = color,
        facecolor = "none"
    )
    ax.add_patch(rect)

    # Title
    check  = "+" if correct  else "x"
    titulo = (
        f"{result['pred_label']}  (idx={result['pred_idx']})\n"
        f"Conf: {result['confidence']*100:.1f}%  |  True: {result['true_label']}  {check}"
    )

    ax.set_title(
        titulo,
        fontsize   = 9,
        color      = color,
        fontweight = "bold",
        pad        = 8,
    )

    ax.set_xticks([])
    ax.set_yticks([])

# Legend
legend_elements = [
    Patch(facecolor=COLORS["plant"],    edgecolor=COLORS["plant"],    label="plant    (idx=0)"),
    Patch(facecolor=COLORS["no_plant"], edgecolor=COLORS["no_plant"], label="no_plant (idx=1)"),
]
fig.legend(
    handles        = legend_elements,
    loc            = "lower center",
    ncol           = 2,
    fontsize       = 11,
    frameon        = True,
    facecolor      = "#1A1A1A",
    edgecolor      = "#444444",
    labelcolor     = "white",
    bbox_to_anchor = (0.5, -0.04)
)

# Accuracy in footer
fig.text(
    0.5, -0.08,
    f"Sample accuracy: {n_correct}/{len(results_data)} ({n_correct/len(results_data)*100:.0f}%)",
    ha         = "center",
    fontsize   = 12,
    color      = "white",
    fontweight = "bold"
)

plt.tight_layout()
output_path = "inference_test_sample.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0D0D0D")
plt.show()
print(f"\nSaved: {output_path}")