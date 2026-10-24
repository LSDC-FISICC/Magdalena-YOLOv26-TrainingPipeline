#!/usr/bin/env python3
"""E3 - How much manual curation is enough? (paper: ablation over [0/10/50/100%]).

Two labellings of the same frames are needed: the pseudo-labelled dataset as
produced by the colour cascade (before anyone touched it) and the curated one
(after manual review). The two Roboflow projects need not share splits or the
exact frame selection: the experiment uses the frames present in both,
matched by source frame name, and re-splits them with the temporal block split
of E0 (test blocks are left out: they are reserved for the GT). For each
curated fraction f and seed, a training set is materialised (symlinks) in which
a fraction f of the frames carries the curated label and the rest the pseudo
label, using every image (augmented copies included) of that frame in either
project. A YOLO26-cls model is trained with the shared protocol
(configs/train.yaml) and evaluated on the independent GT.

Curation strategies:
  random      frames to review are drawn at random (the cost model of the paper),
  confidence  frames are reviewed in order of model uncertainty |p(plant)-0.5|,
              scored by a model trained at f=0 (--scorer); this is "selective
              curation" vs "total curation" (gap G3.2 of the analysis).

Also reports the label-noise rate of the cascade (pseudo vs curated flips),
overall and per class.

Examples:
  python experiments/e3_ablation_curation.py --pseudo roboflow/sugarcane_exg_hsv_lab_v3 \
      --curated roboflow/sugarcane_exg_hsv_lab_v3-2-100 --gt experiments/gt/gt.csv --fractions 0,10,25,50,100 --seeds 0,1,2
  python experiments/e3_ablation_curation.py ... --dry-run          # only build sets + noise stats
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp.data import block_split, link_or_copy, list_folder_dataset, read_csv, write_csv  # noqa: E402
from mexp.metrics import summary  # noqa: E402


def index(rows):
    """source -> {"label": l, "frame": n, "files": [paths]} over all splits of a project."""
    out = {}
    for r in rows:
        e = out.setdefault(r["source"], {"label": r["label"], "frame": r["frame"], "files": []})
        e["files"].append(r["path"])
    return out


def order_for_review(sources, strategy, rng, scores=None):
    sources = sorted(sources)
    if strategy == "random":
        rng.shuffle(sources)
        return sources
    missing = [s for s in sources if s not in scores]
    ranked = sorted((s for s in sources if s in scores), key=lambda s: abs(scores[s] - 0.5))
    rng.shuffle(missing)
    return ranked + missing


def build_set(pseudo, curated, assign, fraction, strategy, seed, dst, scores=None):
    """Materialise train/valid for one curated fraction; returns per-split label counts."""
    rng = random.Random(seed)
    shared = [s for s in pseudo if s in curated and assign.get(pseudo[s]["frame"]) in ("train", "valid")]
    order = order_for_review(shared, strategy, rng, scores)
    reviewed = set(order[:round(fraction / 100 * len(order))])
    n = Counter()
    for s in shared:
        split = assign[pseudo[s]["frame"]]
        label = curated[s]["label"] if s in reviewed else pseudo[s]["label"]
        files = {Path(f).name: f for f in pseudo[s]["files"] + curated[s]["files"]}
        if split != "train":
            files = dict([sorted(files.items())[0]])
        for name, f in files.items():
            link_or_copy(f, Path(dst) / split / label / name)
        n[(split, label)] += 1
        n[(split, "reviewed")] += int(s in reviewed)
        n[(split, "flipped")] += int(s in reviewed and curated[s]["label"] != pseudo[s]["label"])
    return {sp: {k[1]: v for k, v in n.items() if k[0] == sp} for sp in ("train", "valid")}


def noise_stats(pseudo, curated):
    shared = [s for s in pseudo if s in curated]
    c = Counter((pseudo[s]["label"], curated[s]["label"]) for s in shared)
    n = len(shared)
    return {"frames_shared": n, "only_in_pseudo": len(pseudo) - n, "only_in_curated": len(curated) - n,
            "flip_rate": round(sum(v for (a, b), v in c.items() if a != b) / max(n, 1), 4),
            "pseudo_plant_to_no_plant": c[("plant", "no_plant")],
            "pseudo_no_plant_to_plant": c[("no_plant", "plant")],
            "agree_plant": c[("plant", "plant")], "agree_no_plant": c[("no_plant", "no_plant")]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pseudo", required=True, help="pseudo-labelled folder dataset (train/valid)")
    ap.add_argument("--curated", required=True, help="curated folder dataset (train/valid)")
    ap.add_argument("--gt", help="GT csv from e1_agreement.py")
    ap.add_argument("--fractions", default="0,10,25,50,100")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--strategy", choices=["random", "confidence"], default="random")
    ap.add_argument("--scorer", help="weights trained at 0%% curation, for --strategy confidence")
    ap.add_argument("--model", default="yolo26n-cls.pt")
    ap.add_argument("--work", default="experiments/work/e3")
    ap.add_argument("--out", default="experiments/results/e3_curation")
    ap.add_argument("--epochs", type=int, help="override configs/train.yaml (e.g. shorter CPU runs)")
    ap.add_argument("--patience", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device", help="0 (GPU) or cpu; default autodetect")
    ap.add_argument("--block-len", type=int, default=300)
    ap.add_argument("--block-gap", type=int, default=30)
    ap.add_argument("--block-seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pseudo = index(list_folder_dataset(args.pseudo))
    curated = index(list_folder_dataset(args.curated))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    noise = noise_stats(pseudo, curated)
    assign = block_split([v["frame"] for v in pseudo.values() if v["frame"] is not None],
                         args.block_len, args.block_gap, seed=args.block_seed)
    (out / "label_noise.json").write_text(json.dumps(noise, indent=2))
    print("Label noise of the pseudo-labeller:", json.dumps(noise, indent=2))

    scores = None
    if args.strategy == "confidence":
        if not args.scorer:
            sys.exit("--strategy confidence needs --scorer (a model trained at 0% curation)")
        from mexp.models import Predictor
        srcs = sorted(pseudo)
        _, p = Predictor(args.scorer).predict([pseudo[s]["files"][0] for s in srcs])
        scores = dict(zip(srcs, p))

    gt = read_csv(args.gt) if args.gt else None
    results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for frac in [float(f) for f in args.fractions.split(",")]:
            name = f"{args.strategy}_f{frac:g}_s{seed}"
            dst = Path(args.work) / name
            done = Path(args.work) / f"{name}.done.json"
            if done.exists():  # finished in an earlier, interrupted invocation
                results.append(json.loads(done.read_text()))
                print(f"[{name}] already done, skipped")
                continue
            counts = build_set(pseudo, curated, assign, frac, args.strategy, seed, dst, scores)
            print(f"[{name}] {counts}")
            row = {"strategy": args.strategy, "fraction": frac, "seed": seed,
                   "train_counts": json.dumps(counts["train"])}
            if not args.dry_run:
                from mexp.models import Predictor
                from mexp.train import load_cfg, train_yolo
                cfg = load_cfg(epochs=args.epochs, patience=args.patience, batch=args.batch)
                best = train_yolo(dst, Path(args.work) / "runs", name, model=args.model, cfg=cfg, seed=seed,
                                  device=args.device)
                row["weights"] = str(best)
                if gt:
                    pred, _ = Predictor(best).predict([r["path"] for r in gt])
                    row.update({k: v for k, v in summary([r["label"] for r in gt], pred).items()})
                done.write_text(json.dumps(row))
            results.append(row)
            write_csv(out / f"runs_{args.strategy}.csv", results, list(dict.fromkeys(k for r in results for k in r)))

    if gt and not args.dry_run:
        agg = []
        for frac in sorted({r["fraction"] for r in results}):
            accs = np.array([r["accuracy"] for r in results if r["fraction"] == frac])
            agg.append({"fraction": frac, "acc_mean": accs.mean(), "acc_std": accs.std(ddof=1) if len(accs) > 1 else 0,
                        "n_seeds": len(accs)})
        write_csv(out / f"summary_{args.strategy}.csv", agg)
        for a in agg:
            print(f"  curated {a['fraction']:>5g}%  acc = {100 * a['acc_mean']:.2f} +- {100 * a['acc_std']:.2f}")


if __name__ == "__main__":
    main()
