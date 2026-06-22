"""
inference_benchmark.py
======================
Benchmarks YOLO26s classification model across three backends:
  - PT     : PyTorch via Ultralytics YOLO  (GPU)
  - ONNX   : ONNXRuntime CPUExecutionProvider  (letterbox + /255)
  - Engine : TensorRT engine via Ultralytics YOLO  (GPU, FP16)

Measures per-image latency, accuracy and confidence over the test set,
then produces a multi-panel comparison plot.

Usage:
    python inference_benchmark.py

Notes:
  - The Engine file is expected at ENGINE_MODEL path.
    Export it first with:
        yolo export model=best.pt format=engine imgsz=320 half=True device=0
  - ONNX uses ONNXRuntime CPU because the installed ORT wheel requires
    libcublasLt.so.12 which is not present on this system (CUDA 13 only).
    The PT and Engine backends both run fully on the GPU.
"""

import random
import time
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as T
import os

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
PT_MODEL     = "runs/classify/magdalena/plant_cls_v4/weights/best.pt"
ONNX_MODEL   = "runs/classify/magdalena/plant_cls_v4/weights/best.onnx"
ENGINE_MODEL = "runs/classify/magdalena/plant_cls_v4/weights/best.engine"
TEST_DIR     = "dataset2/test"
N_IMAGES     = 50         # images used for benchmarking
WARMUP_RUNS  = 10         # warm-up passes per backend (not counted in timings)
SEED         = 42
IMGSZ        = 320        # model square input size (px)
DEVICE_IDX   = 0          # GPU index for PT / Engine backends
OUTPUT_PATH  = "inference_benchmark.png"
BATCH_SIZE = 5
# ─────────────────────────────────────────────────────────────────────────────

#no_present normally
INFER_TRANSFORM = T.Compose([
    T.Resize((96, 320)),
    T.ToTensor(),
])



PALETTE = {
    "PT":     "#4FC3F7",   # light blue
    "ONNX":   "#81C784",   # green
    "Engine": "#FFB74D",   # orange
}
BG_COLOR   = "#0D0D0D"
GRID_COLOR = "#2A2A2A"
TEXT_COLOR = "white"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def collect_images(test_dir: str, n: int, seed: int) -> list[dict]:
    test_path = Path(test_dir)
    all_images = []
    for class_dir in sorted(test_path.iterdir()):
        if class_dir.is_dir():
            for img_file in class_dir.glob("*"):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    all_images.append({"path": img_file, "true_label": class_dir.name})
    random.seed(seed)
    return random.sample(all_images, min(n, len(all_images)))


def letterbox(img: Image.Image, target: int = 320, pad_val: int = 114) -> Image.Image:
    """Resize maintaining aspect ratio then pad with grey to a square canvas."""
    w, h = img.size
    scale = target / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), (pad_val, pad_val, pad_val))
    canvas.paste(img, ((target - nw) // 2, (target - nh) // 2))
    return canvas


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def print_summary(name: str, results: list[dict]) -> None:
    n_correct = sum(r["correct"] for r in results)
    lat = [r["latency_ms"] for r in results]
    conf = [r["confidence"] for r in results]
    print(f"\n{'─'*65}")
    print(f"  {name}")
    print(f"{'─'*65}")
    print(f"  Accuracy    : {n_correct}/{len(results)} ({n_correct/len(results)*100:.1f}%)")
    print(f"  Latency     : {np.mean(lat):.2f} ± {np.std(lat):.2f} ms  "
          f"[min={np.min(lat):.2f}  max={np.max(lat):.2f}]")
    print(f"  Throughput  : {1000/np.mean(lat):.1f} img/s")
    print(f"  Confidence  : {np.mean(conf):.3f} ± {np.std(conf):.3f}")


# ─── Backend: PyTorch (Ultralytics YOLO) ─────────────────────────────────────

def run_pt(samples: list[dict], class_names: dict) -> dict:
    print(f"\n[PT] Loading {PT_MODEL} ...")
    model = YOLO(PT_MODEL)

    all_images = [(s["true_label"], Path(s["path"]).name, s["path"]) for s in samples]

    dummy = [all_images[0][2]] * BATCH_SIZE

    for _ in range(WARMUP_RUNS):
        model.predict(source=dummy, imgsz=[96,320], device=DEVICE_IDX, verbose=False)

    results = []
    batch_times = []

    for batch_idx, batch_start in enumerate(range(0, len(all_images), BATCH_SIZE)):
        batch = all_images[batch_start:batch_start + BATCH_SIZE]
        batch_paths = [item[2] for item in batch]
        t0 = time.perf_counter()
        res = model.predict(batch_paths, imgsz = [96,320], device=DEVICE_IDX, verbose=False)
        t1 = time.perf_counter()

        for (true_label, img_name, img_path), res in zip(batch, res):
            if hasattr(res, 'probs') and res.probs is not None:
                pred_idx = res.probs.top1
                pred_label = res.names[pred_idx]
                confidence = float(res.probs.top1conf)
            else:
                print(f"⚠ Unexpected result format for {img_name}")
                continue

            if pred_label == 'class0':
                pred_label = 'no_plant'
            elif pred_label == 'class1':
                pred_label = 'plant'

            results.append({
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "correct":    pred_label == true_label,
            })

        batch_times.append(
            (t1 - t0) * 1000
        )

    #print_summary("PT (GPU — PyTorch)", results)
    return {"predictions": results, "batch_times": batch_times}


# ─── Backend: ONNX (ORT + CPU) ───────────────────────────────────────────────

def run_onnx(samples: list[dict], class_names: dict) -> dict:

    onnx_path = Path(ONNX_MODEL)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"Engine file not found: {ONNX_MODEL}\n"
            "Export it first:\n"
            "  yolo export model=best.pt format=onnx imgsz=320 half=True device=0"
        )

    print(f"\n[Engine] Loading {ONNX_MODEL} (TensorRT FP16) ...")
    model = YOLO(ONNX_MODEL, task ='classify')

    all_images = [(s["true_label"], Path(s["path"]).name, s["path"]) for s in samples]

    dummy = [all_images[0][2]] * BATCH_SIZE

    for _ in range(WARMUP_RUNS):
        model.predict(source=dummy, imgsz=[96,320], device=DEVICE_IDX, verbose=False)

    results = []
    batch_times = []

    for batch_idx, batch_start in enumerate(range(0, len(all_images), BATCH_SIZE)):
        batch = all_images[batch_start:batch_start + BATCH_SIZE]
        batch_paths = [item[2] for item in batch]
        t0 = time.perf_counter()
        res = model.predict(batch_paths, imgsz = [96,320], device=DEVICE_IDX, verbose=False)
        t1 = time.perf_counter()

        for (true_label, img_name, img_path), res in zip(batch, res):
            if hasattr(res, 'probs') and res.probs is not None:
                pred_idx = res.probs.top1
                pred_label = res.names[pred_idx]
                confidence = float(res.probs.top1conf)
            else:
                print(f"⚠ Unexpected result format for {img_name}")
                continue

            if pred_label == 'class0':
                pred_label = 'no_plant'
            elif pred_label == 'class1':
                pred_label = 'plant'



            results.append({
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "correct":    pred_label == true_label,
            })

        batch_times.append(
            (t1 - t0) * 1000
        )
    #print_summary("ONNX (CPU — ONNXRuntime)", results)
    return {"predictions": results, "batch_times": batch_times}


# ─── Backend: TensorRT Engine (Ultralytics YOLO + TRT) ───────────────────────

def run_engine(samples: list[dict], class_names: dict) -> dict:
    engine_path = Path(ENGINE_MODEL)
    if not engine_path.exists():
        raise FileNotFoundError(
            f"Engine file not found: {ENGINE_MODEL}\n"
            "Export it first:\n"
            "  yolo export model=best.pt format=engine imgsz=320 half=True device=0"
        )

    print(f"\n[Engine] Loading {ENGINE_MODEL} (TensorRT FP16) ...")
    model = YOLO(ENGINE_MODEL)

    all_images = [(s["true_label"], Path(s["path"]).name, s["path"]) for s in samples]

    dummy = [all_images[0][2]] * BATCH_SIZE

    for _ in range(WARMUP_RUNS):
        model.predict(source=dummy, imgsz=[96,320], device=DEVICE_IDX, verbose=False)

    results = []
    batch_times = []

    for batch_idx, batch_start in enumerate(range(0, len(all_images), BATCH_SIZE)):
        batch = all_images[batch_start:batch_start + BATCH_SIZE]
        batch_paths = [item[2] for item in batch]
        t0 = time.perf_counter()
        res = model.predict(batch_paths, imgsz = [96,320], device=DEVICE_IDX, verbose=False)
        t1 = time.perf_counter()

        for (true_label, img_name, img_path), res in zip(batch, res):
            if hasattr(res, 'probs') and res.probs is not None:
                pred_idx = res.probs.top1
                pred_label = res.names[pred_idx]
                confidence = float(res.probs.top1conf)
            else:
                print(f"⚠ Unexpected result format for {img_name}")
                continue

            if pred_label == 'class0':
                pred_label = 'no_plant'
            elif pred_label == 'class1':
                pred_label = 'plant'


            results.append({
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "correct":    pred_label == true_label,
            })

        batch_times.append(
            (t1 - t0) * 1000
        )

    #print_summary("Engine (GPU — TensorRT FP16)", results)
    return {"predictions": results, "batch_times": batch_times}


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_benchmark(backend_results: dict[str, list[dict]], samples: list[dict]) -> None:
    backends = list(backend_results.keys())
    colors   = [PALETTE[b] for b in backends]

    latencies   = {b: v["batch_times"] for b, v in backend_results.items()}
    accuracies  = {b: np.mean([r["correct"]    for r in v["predictions"]]) * 100 for b, v in backend_results.items()}
    confidences = {b: [r["confidence"] for r in v["predictions"]]             for b, v in backend_results.items()}
    throughputs = {b: BATCH_SIZE * 1000 / np.mean(latencies[b])               for b in backends}
    means       = {b: np.mean(latencies[b])                                   for b in backends}
    stds        = {b: np.std(latencies[b])                                    for b in backends}
    pt_mean     = means["PT"]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 15), facecolor=BG_COLOR)
    fig.suptitle(
        "Inference Benchmark  —  PT (GPU)  vs  ONNX (CPU)  vs  TensorRT Engine (GPU, FP16)\n"
        "Proyecto Magdalena  |  YOLO26s Classification  |  RTX 2060 Laptop",
        fontsize=15, fontweight="bold", color=TEXT_COLOR, y=0.99
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.55,
        wspace=0.38,
        left=0.07, right=0.97,
        top=0.92, bottom=0.08
    )

    ax_mean   = fig.add_subplot(gs[0, 0])
    ax_tput   = fig.add_subplot(gs[0, 1])
    ax_acc    = fig.add_subplot(gs[0, 2])
    ax_box    = fig.add_subplot(gs[1, :2])
    ax_violin = fig.add_subplot(gs[1, 2])
    ax_line   = fig.add_subplot(gs[2, :])

    def style_ax(ax, title, xlabel="", ylabel=""):
        ax.set_facecolor("#161616")
        ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, linestyle="--")
        ax.set_axisbelow(True)

    x     = np.arange(len(backends))
    bar_w = 0.5

    # Labels with device annotation
    x_labels = {
        "PT":     "PT\n(GPU)",
        "ONNX":   "ONNX\n(CPU)",
        "Engine": "Engine\n(GPU FP16)",
    }

    # ── Mean latency ──────────────────────────────────────────────────────────
    style_ax(ax_mean, f"Mean Latency (batch={BATCH_SIZE})", ylabel="ms / batch")
    bars = ax_mean.bar(x, [means[b] for b in backends], bar_w, color=colors,
                       yerr=[stds[b] for b in backends], capsize=5,
                       error_kw={"ecolor": "white", "alpha": 0.7})
    ax_mean.set_xticks(x)
    ax_mean.set_xticklabels([x_labels[b] for b in backends], color=TEXT_COLOR, fontsize=8)
    for bar, b in zip(bars, backends):
        ax_mean.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + stds[b] + ax_mean.get_ylim()[1] * 0.01,
                     f"{means[b]:.2f} ms", ha="center", va="bottom",
                     color=TEXT_COLOR, fontsize=8, fontweight="bold")
    for bar, b in zip(bars, backends):
        if b != "PT":
            spd = pt_mean / means[b]
            ax_mean.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 0.45,
                         f"{spd:.2f}×", ha="center", va="center",
                         color="black", fontsize=8, fontweight="bold")

    # ── Throughput ────────────────────────────────────────────────────────────
    style_ax(ax_tput, "Throughput", ylabel="images / second")
    bars_t = ax_tput.bar(x, [throughputs[b] for b in backends], bar_w, color=colors)
    ax_tput.set_xticks(x)
    ax_tput.set_xticklabels([x_labels[b] for b in backends], color=TEXT_COLOR, fontsize=8)
    for bar, b in zip(bars_t, backends):
        ax_tput.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + ax_tput.get_ylim()[1] * 0.01,
                     f"{throughputs[b]:.0f}", ha="center", va="bottom",
                     color=TEXT_COLOR, fontsize=8, fontweight="bold")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    style_ax(ax_acc, "Accuracy", ylabel="%")
    ax_acc.set_ylim(0, 113)
    bars_a = ax_acc.bar(x, [accuracies[b] for b in backends], bar_w, color=colors)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([x_labels[b] for b in backends], color=TEXT_COLOR, fontsize=8)
    ax_acc.axhline(100, color="white", linewidth=0.5, linestyle="--", alpha=0.3)
    for bar, b in zip(bars_a, backends):
        ax_acc.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{accuracies[b]:.1f}%", ha="center", va="bottom",
                    color=TEXT_COLOR, fontsize=8, fontweight="bold")

    # ── Latency box plot ──────────────────────────────────────────────────────
    style_ax(ax_box, f"Latency Distribution (batch={BATCH_SIZE})", ylabel="ms / batch")
    bp = ax_box.boxplot(
        [latencies[b] for b in backends],
        labels=[x_labels[b] for b in backends],
        patch_artist=True,
        notch=False,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": TEXT_COLOR},
        capprops={"color": TEXT_COLOR},
        flierprops={"markerfacecolor": "gray", "markersize": 3, "alpha": 0.5},
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for label in ax_box.get_xticklabels():
        label.set_color(TEXT_COLOR)
    rng = np.random.default_rng(0)
    for i, b in enumerate(backends, start=1):
        jitter = rng.uniform(-0.15, 0.15, len(latencies[b]))
        ax_box.scatter(np.full(len(latencies[b]), i) + jitter, latencies[b],
                       color=PALETTE[b], alpha=0.35, s=12, zorder=3)

    # ── Confidence violin ─────────────────────────────────────────────────────
    style_ax(ax_violin, "Confidence Distribution", ylabel="confidence score")
    parts = ax_violin.violinplot(
        [confidences[b] for b in backends],
        positions=range(len(backends)),
        showmedians=True,
        showextrema=True,
    )
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("white")
    ax_violin.set_xticks(range(len(backends)))
    ax_violin.set_xticklabels([x_labels[b] for b in backends], color=TEXT_COLOR, fontsize=8)
    ax_violin.set_ylim(0, 1.08)

    # ── Per-image latency line ────────────────────────────────────────────────
    style_ax(ax_line, f"Per-Batch Latency (batch={BATCH_SIZE})",
             xlabel="Batch index", ylabel="ms / batch")
    for b in backends:
        idx = np.arange(len(latencies[b]))
        ax_line.plot(idx, latencies[b], color=PALETTE[b],
                     linewidth=1.2, alpha=0.85, label=x_labels[b].replace("\n", " "))
        ax_line.fill_between(idx, latencies[b], alpha=0.12, color=PALETTE[b])

    ax_line.legend(facecolor="#1A1A1A", edgecolor=GRID_COLOR,
                   labelcolor=TEXT_COLOR, fontsize=9)

    # ── Footer stats table ────────────────────────────────────────────────────
    stat_lines = []
    for b in backends:
        n_ok = sum(r["correct"] for r in backend_results[b]["predictions"])
        n    = len(backend_results[b])
        spd  = f"   ({pt_mean/means[b]:.2f}× vs PT)" if b != "PT" else ""
        stat_lines.append(
            f"{b:<8}  lat={means[b]:.2f}±{stds[b]:.2f} ms  "
            f"tput={throughputs[b]:.0f} img/s  "
            f"acc={n_ok}/{n} ({accuracies[b]:.1f}%)  "
            f"conf={np.mean(confidences[b]):.3f}{spd}"
        )
    fig.text(0.5, 0.015, "\n".join(stat_lines),
             ha="center", va="bottom", fontsize=8,
             color="#AAAAAA", family="monospace")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [mpatches.Patch(color=PALETTE[b], label=x_labels[b].replace("\n", " "))
                      for b in backends]
    fig.legend(handles=legend_patches, loc="upper right",
               bbox_to_anchor=(0.97, 0.97),
               facecolor="#1A1A1A", edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"\nPlot saved → {OUTPUT_PATH}")
    plt.show()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 65)
    print("  Inference Benchmark  —  Proyecto Magdalena  (YOLO26s)")
    print("=" * 65)
    print(f"  CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU            : {torch.cuda.get_device_name(DEVICE_IDX)}")

    _model = YOLO(PT_MODEL)
    class_names: dict = _model.names
    print(f"  Classes        : {class_names}")
    del _model

    samples = collect_images(TEST_DIR, N_IMAGES, SEED)
    print(f"  Benchmark set  : {len(samples)} images  (warmup={WARMUP_RUNS})\n")
    print(f"  Backends:")
    print(f"    PT     → {PT_MODEL}")
    print(f"    ONNX   → {ONNX_MODEL}")
    print(f"    Engine → {ENGINE_MODEL}")

    backend_results: dict[str, dict[str, list]] = {}
    backend_results["PT"]     = run_pt(samples, class_names)
    backend_results["ONNX"]   = run_onnx(samples, class_names)
    backend_results["Engine"] = run_engine(samples, class_names)

    # ── Final comparison table ────────────────────────────────────────────────
    pt_mean = np.mean(backend_results["PT"]["batch_times"])
    print("\n" + "=" * 65)
    print("  FINAL COMPARISON")
    print("=" * 65)
    header = f"  {'Backend':<10} {'Mean (ms)':>10} {'Std (ms)':>9} {'Acc':>10} {'Tput (img/s)':>13} {'Speedup':>9}"
    print(header)
    print("  " + "─" * 63)
    for b, res in backend_results.items():
        lats  = res["batch_times"]
        preds = res["predictions"]
        n_ok = sum(r["correct"] for r in preds)
        n     = len(res)
        spd   = f"{pt_mean/np.mean(lats):.2f}×" if b != "PT" else "baseline"
        print(f"  {b:<10} {np.mean(lats):>10.2f} {np.std(lats):>9.2f} "
              f"{n_ok}/{n} ({n_ok/n*100:>4.1f}%) {1000/np.mean(lats):>13.0f} {spd:>9}")
    print("=" * 65)

    plot_benchmark(backend_results, samples)


if __name__ == "__main__":
    main()
