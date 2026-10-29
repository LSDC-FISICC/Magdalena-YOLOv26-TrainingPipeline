#!/usr/bin/env python3
"""E12 - Add pseudo-labelled night and blurred frames to the training set.

The daylight model calls bare soil "cane" in blurred and night passes. Here unlabelled frames of
those passes are labelled automatically and added to the curated daylight training set, and the
model (YOLO26n-cls, E11 online augmentation) is retrained.

  label  pseudo-labels every --stride-th frame of each source group with two independent
         labellers, the YOLO model trained on ExG labels (--labeler) and the HSV rule, and keeps
         only the frames on which they agree (the rest are dropped as uncertain). Writes
         <out>/<group>/<plant|no_plant>/*.jpg and <out>/pseudo.csv.
  run    for every fold, trains on the curated set + the pseudo frames of the fold's training
         groups, and scores the human GT. A GT image that belongs to a group (night video, half
         of a recording) is scored only by the folds that did not train on that group, so no
         model is tested on a pass it saw; images without a group (daylight GT, whose frames
         are outside the training blocks) are scored by every fold.

Example (groups and folds used for the paper):
  python experiments/e12_pseudo_finetune.py label --labeler exg_v1_best.pt --out experiments/e12_data \\
      --source night_0515:/data/night_crops/mid/noche_2025-05-15:2 \\
      --source night_0603:/data/night_crops/mid/noche_2025-06-03:2 \\
      --source s2_a:/data/rec2:1:0:1374 --source s2_b:/data/rec2:1:1375:99999
  python experiments/e12_pseudo_finetune.py run --data experiments/e12_data \\
      --curated experiments/work/gpu/data/curated --folds gt_folds.csv \\
      --fold A=night_0603,s2_a --fold B=night_0515,s2_b \\
      --gt day=experiments/gt_human/dia/gt.csv --gt session2=experiments/gt_human/sesion2/gt.csv \\
      --gt night=experiments/gt_human/noche/gt.csv --seeds 0,1,2 --device 0

gt_folds.csv maps (domain, image_id) to the group of that GT image. The deployed model is then
trained on every group (--fold ALL=night_0515,night_0603,s2_a,s2_b); only the daylight GT, which
no fold trains on, scores it, and run_gpu_suite.py deploy --main e12 picks its seed.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp import REPO_ROOT  # noqa: E402
from mexp.data import block_of, frame_id, link_or_copy, list_folder_dataset, read_csv, write_csv  # noqa: E402
from mexp.metrics import summary  # noqa: E402


def parse_source(spec):
    """name:dir[:stride[:first:last]]"""
    f = spec.split(":")
    stride = int(f[2]) if len(f) > 2 else 1
    first, last = (int(f[3]), int(f[4])) if len(f) > 4 else (0, 10 ** 9)
    return f[0], Path(f[1]), stride, first, last


def label(args):
    from mexp import pseudolabel
    from mexp.models import Predictor
    labeler = Predictor(args.labeler)
    out = Path(args.out)
    rows = []
    for spec in args.source:
        name, d, stride, first, last = parse_source(spec)
        frames = sorted((frame_id(p), p) for p in d.glob("*.png") if first <= frame_id(p) <= last)[::stride]
        paths = [str(p) for _, p in frames]
        model, _ = labeler.predict(paths)
        kept = 0
        for (fr, p), m in zip(frames, model):
            h = pseudolabel.hsv(p)
            agree = m == h
            rows.append({"group": name, "frame": fr, "src": str(p), "model": m, "hsv": h,
                         "label": m if agree else ""})
            if agree:
                dst = out / name / m / f"{name}_frame_{fr:05d}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), cv2.imread(str(p)), [cv2.IMWRITE_JPEG_QUALITY, 95])
                kept += 1
        n_plant = sum(r["label"] == "plant" for r in rows if r["group"] == name)
        print(f"{name}: {len(frames)} frames, {kept} kept ({100 * kept / max(len(frames), 1):.0f}%), "
              f"{n_plant} plant / {kept - n_plant} no_plant")
    write_csv(out / "pseudo.csv", rows)


def build_fold(data, curated, train_groups, dst):
    """curated train/valid + pseudo frames of train_groups (valid: frames of the validation blocks)."""
    for split in ("train", "valid"):
        for r in list_folder_dataset(curated, split):
            link_or_copy(r["path"], dst / split / r["label"] / Path(r["path"]).name)
    n = defaultdict(int)
    for g in train_groups:
        for r in list_folder_dataset(data / g):
            split = "valid" if block_of(r["frame"]) == "valid" else "train"
            link_or_copy(r["path"], dst / split / r["label"] / Path(r["path"]).name)
            n[split] += 1
    return dict(n)


def run(args):
    from mexp import tv_models
    from mexp.models import Predictor
    from mexp.train import load_cfg, train_yolo
    data, work, out = Path(args.data), Path(args.work), Path(args.out)
    aname = Path(args.arch).stem
    cfg = load_cfg(epochs=args.epochs, patience=args.patience, batch=args.batch)
    folds = {k: v.split(",") for k, v in (f.split("=", 1) for f in args.fold)}
    group_of = {(r["domain"], r["image_id"]): r["group"] for r in read_csv(args.folds)}
    gts = {k: read_csv(v) for k, v in (g.split("=", 1) for g in args.gt)}
    records = []
    for fold, train_groups in folds.items():
        ddir = work / "data" / f"fold_{fold}"
        n = build_fold(data, Path(args.curated), train_groups, ddir)
        for seed in [int(s) for s in args.seeds.split(",")]:
            run_name = f"fold{fold}__{aname}_aug_pseudo__s{seed}"
            done = work / f"{run_name}.done.json"
            if done.exists():
                records.append(json.loads(done.read_text()))
                continue
            print(f"\n=== {run_name}: pseudo frames {n}, train groups {train_groups}")
            if args.arch in tv_models.ARCHS:
                best = tv_models.train(args.arch, ddir, work / "runs" / run_name, epochs=cfg["epochs"],
                                       patience=cfg["patience"], batch=cfg["batch"], seed=seed,
                                       device=args.device, augment=True)
            else:
                best = train_yolo(ddir, work / "runs", run_name, model=args.arch, cfg=cfg, seed=seed,
                                  device=args.device, aug=True)
            pred = Predictor(best)
            rec = {"fold": fold, "seed": seed, "arch": aname, "weights": str(best), "pseudo": n,
                   "preds": {}, "probs": {}}
            for dom, gt in gts.items():
                test = [r for r in gt if group_of.get((dom, r["image_id"])) not in train_groups]
                if not test:  # e.g. the final model trained on every group
                    continue
                labels, probs = pred.predict([str(REPO_ROOT / r["path"]) if not Path(r["path"]).is_absolute()
                                              else r["path"] for r in test])
                rec["preds"][dom] = {r["image_id"]: p for r, p in zip(test, labels)}
                rec["probs"][dom] = {r["image_id"]: float(q) for r, q in zip(test, probs)}
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text(json.dumps(rec))
            records.append(rec)
    # Per seed: every grouped GT image scored once (by the fold that did not train on its group);
    # ungrouped images (day) scored by every fold and pooled.
    rows = []
    for seed in sorted({r["seed"] for r in records}):
        recs = [r for r in records if r["seed"] == seed]
        row = {"seed": seed}
        for dom, gt in gts.items():
            truth = {r["image_id"]: r["label"] for r in gt}
            y, p = [], []
            for r in recs:
                probs = r.get("probs", {}).get(dom)
                for iid, lab in r["preds"].get(dom, {}).items():
                    if iid not in truth:  # e.g. a repeated image left out of a deduplicated GT
                        continue
                    if probs is not None:  # --threshold applies when probabilities were stored
                        lab = "plant" if probs[iid] >= args.threshold else "no_plant"
                    y.append(truth[iid]); p.append(lab)
            if not y:
                continue
            s = summary(y, p)
            row.update({f"{dom}_acc": round(s["accuracy"], 4), f"{dom}_recall": round(s["recall"], 4),
                        f"{dom}_spec": round(s["specificity"], 4), f"{dom}_n": len(y)})
        rows.append(row)
        print(row)
    write_csv(out / "runs.csv", rows)
    if len(rows) > 1:
        mean = {"seed": "mean"}
        std = {"seed": "std"}
        for k in rows[0]:
            if k != "seed":
                v = np.array([r[k] for r in rows], float)
                mean[k], std[k] = round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4)
        write_csv(out / "summary.csv", [mean, std])
        print("mean", mean, "\nstd ", std)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["label", "run"])
    ap.add_argument("--source", action="append", help="label: name:dir[:stride[:first:last]]")
    ap.add_argument("--labeler", help="label: YOLO weights trained on ExG labels")
    ap.add_argument("--out", default="experiments/results/e12", help="label: pseudo set dir; run: results dir")
    ap.add_argument("--data", default="experiments/e12_data", help="run: pseudo set from the label step")
    ap.add_argument("--curated", help="run: block-split curated dataset (train/valid)")
    ap.add_argument("--folds", help="run: csv domain,image_id,group of the GT images")
    ap.add_argument("--fold", action="append", help="run: NAME=group,group (training groups of the fold)")
    ap.add_argument("--gt", action="append", help="run: domain=gt.csv")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--arch", default="yolo26n-cls.pt",
                    help="run: YOLO weights or a torchvision arch (efficientnet_b0, mobilenet_v3_small, ...)")
    ap.add_argument("--threshold", type=float, default=0.5, help="run: p(plant) threshold when scoring")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device")
    ap.add_argument("--work", default="experiments/work/e12")
    args = ap.parse_args()
    {"label": label, "run": run}[args.step](args)


if __name__ == "__main__":
    main()
