#!/usr/bin/env python3
"""E4b/E5 - Train under one protocol and compare on the GT.

Two uses of the same driver:

E4  pseudo-label source: the same architecture (YOLO26s-cls) trained on each
    Roboflow variant (HSV v1..v5, ExG v1, ExG+HSV v1..v2, ExG+HSV+Lab v1..v3,
    manual SugarCaneV1). Supports the highlight "Cascaded HSV, LAB and ExGR
    masks replace most manual image labeling".
E5  architecture: the same dataset with YOLO26n/s-cls and torchvision
    MobileNetV3-Small/Large, EfficientNet-B0, ResNet18 (weakness D6 of the
    analysis). Reports parameters and GFLOPs at 96x320 next to accuracy;
    latency of each model on the Jetson comes from e6_benchmark_latency.py
    using the exported .onnx/.engine.

Every --dataset is re-split with the temporal block split (--block-split) so
that the held-out blocks, and therefore the GT, are the same for all variants.

Examples:
  # E4
  python experiments/e4_train_compare.py --gt experiments/gt/gt.csv --block-split \
      --dataset hsv_v4=roboflow/sugarcane_hsv_v4 --dataset cascade_v3=roboflow/sugarcane_exg_hsv_lab_v3 ...
  # E5
  python experiments/e4_train_compare.py --gt experiments/gt/gt.csv --dataset curated=dataset2_block \
      --arch yolo26n-cls.pt --arch yolo26s-cls.pt --arch mobilenet_v3_small --arch efficientnet_b0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp.data import list_folder_dataset, read_csv, resplit, write_csv  # noqa: E402
from mexp.metrics import summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--arch", action="append", metavar="WEIGHTS_OR_TV_ARCH",
                    help="yolo26*-cls.pt or a torchvision arch (default yolo26s-cls.pt)")
    ap.add_argument("--gt", help="GT csv")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--block-split", action="store_true")
    ap.add_argument("--block-len", type=int, default=300)
    ap.add_argument("--block-gap", type=int, default=30)
    ap.add_argument("--block-seed", type=int, default=0)
    ap.add_argument("--export-onnx", action="store_true", help="export each best model for e6")
    ap.add_argument("--epochs", type=int, help="override configs/train.yaml (e.g. shorter CPU runs)")
    ap.add_argument("--patience", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device")
    ap.add_argument("--work", default="experiments/work/e4")
    ap.add_argument("--out", default="experiments/results/e4_compare")
    args = ap.parse_args()

    from mexp import tv_models
    from mexp.models import Predictor
    from mexp.train import load_cfg, train_yolo

    archs = args.arch or ["yolo26s-cls.pt"]
    gt = read_csv(args.gt) if args.gt else None
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(epochs=args.epochs, patience=args.patience, batch=args.batch)
    rows = []
    for spec in args.dataset:
        dname, ddir = spec.split("=", 1)
        if args.block_split:
            ddir = resplit(ddir, Path(args.work) / "data" / dname, args.block_len, args.block_gap, args.block_seed)
        counts = {sp: len(list_folder_dataset(ddir, sp)) for sp in ("train", "valid")}
        for arch in archs:
            for seed in [int(s) for s in args.seeds.split(",")]:
                aname = Path(arch).stem
                run = f"{dname}__{aname}__s{seed}"
                done = Path(args.work) / f"{run}.done.json"
                if done.exists():  # resume: runs are long on CPU
                    rows.append(json.loads(done.read_text()))
                    write_csv(out / "runs.csv", rows, list(dict.fromkeys(k for r in rows for k in r)))
                    continue
                print(f"\n=== {run}  {counts}")
                if arch in tv_models.ARCHS:
                    best = tv_models.train(arch, ddir, Path(args.work) / "runs" / run, seed=seed,
                                           epochs=cfg["epochs"], patience=cfg["patience"],
                                           batch=cfg["batch"], device=args.device)
                    params, gflops = tv_models.complexity(tv_models.load_checkpoint(best))
                    if args.export_onnx:
                        tv_models.export_onnx(best, str(best.with_suffix(".onnx")))
                else:
                    best = train_yolo(ddir, Path(args.work) / "runs", run, model=arch, cfg=cfg, seed=seed,
                                      device=args.device)
                    from ultralytics import YOLO
                    y = YOLO(str(best))
                    params, gflops = tv_models.complexity(y.model)
                    if args.export_onnx:
                        y.export(format="onnx", imgsz=[96, 320], dynamic=True)
                row = {"dataset": dname, "arch": aname, "seed": seed, "train_images": counts["train"],
                       "params_M": round(params, 3), "gflops_96x320": gflops, "weights": str(best)}
                if gt:
                    pred, _ = Predictor(best).predict([r["path"] for r in gt])
                    row.update(summary([r["label"] for r in gt], pred))
                rows.append(row)
                done.write_text(json.dumps(row, default=str))
                write_csv(out / "runs.csv", rows, list(dict.fromkeys(k for r in rows for k in r)))
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
