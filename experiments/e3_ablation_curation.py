#!/usr/bin/env python3
"""E3 - How much manual curation is enough? (paper: ablation over [0/10/50/100%]).

Two versions of the same frames are needed: the pseudo-labelled dataset as
produced by the colour cascade (before anyone touched it) and the curated one
(after manual review). For each curated fraction f and seed, a training set is
materialised (symlinks) in which a fraction f of the source frames carries the
curated label and the rest the pseudo label; frames that curation deleted are
dropped only if they fall in the curated fraction. A YOLO26-cls model is
trained with the shared protocol (configs/train.yaml) and evaluated on the
independent GT.

Curation strategies:
  random      frames to review are drawn at random (the cost model of the paper),
  confidence  frames are reviewed in order of model uncertainty |p(plant)-0.5|,
              scored by a model trained at f=0 (--scorer); this is "selective
              curation" vs "total curation" (gap G3.2 of the analysis).

Also reports the label-noise rate of the cascade (pseudo vs curated flips),
overall and per class.

Examples:
  python experiments/e3_ablation_curation.py --pseudo roboflow/sugarcane_exg_hsv_lab_v3 \
      --curated dataset2 --gt experiments/gt/gt.csv --fractions 0,10,25,50,100 --seeds 0,1,2
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
from mexp.data import link_or_copy, list_folder_dataset, read_csv, unique_sources, write_csv  # noqa: E402
from mexp.metrics import summary  # noqa: E402


def index(rows):
    """source -> {"label": l, "files": [paths]} per split."""
    out = {}
    for r in rows:
        e = out.setdefault((r["split"], r["source"]), {"label": r["label"], "files": []})
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


def build_set(pseudo, curated, split, fraction, strategy, seed, dst, scores=None):
    rng = random.Random(seed)
    ps = {s: v for (sp, s), v in pseudo.items() if sp == split}
    cu = {s: v for (sp, s), v in curated.items() if sp == split}
    sources = set(ps) | set(cu)
    order = order_for_review(sources, strategy, rng, scores)
    reviewed = set(order[:round(fraction / 100 * len(order))])
    n = Counter()
    for s in sources:
        if s in reviewed:
            if s not in cu:
                n["dropped_by_curation"] += 1
                continue
            label, files = cu[s]["label"], cu[s]["files"]
        else:
            if s not in ps:
                continue
            label, files = ps[s]["label"], ps[s]["files"]
        for f in files:
            link_or_copy(f, Path(dst) / split / label / Path(f).name)
        n[label] += 1
    n["reviewed"] = len(reviewed)
    return dict(n)


def noise_stats(pseudo, curated):
    both = [(pseudo[k]["label"], curated[k]["label"]) for k in pseudo if k in curated]
    removed = sum(1 for k in pseudo if k not in curated)
    c = Counter(both)
    n = len(both)
    return {"frames_in_both": n, "removed_by_curation": removed,
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
    ap.add_argument("--model", default="yolo26s-cls.pt")
    ap.add_argument("--work", default="experiments/work/e3")
    ap.add_argument("--out", default="experiments/results/e3_curation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pseudo = index(list_folder_dataset(args.pseudo))
    curated = index(list_folder_dataset(args.curated))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    noise = {sp: noise_stats({k: v for k, v in pseudo.items() if k[0] == sp},
                             {k: v for k, v in curated.items() if k[0] == sp}) for sp in ("train", "valid")}
    (out / "label_noise.json").write_text(json.dumps(noise, indent=2))
    print("Label noise of the pseudo-labeller:", json.dumps(noise, indent=2))

    scores = None
    if args.strategy == "confidence":
        if not args.scorer:
            sys.exit("--strategy confidence needs --scorer (a model trained at 0% curation)")
        from mexp.models import Predictor
        rows = unique_sources([r for r in list_folder_dataset(args.pseudo) if r["split"] in ("train", "valid")])
        _, p = Predictor(args.scorer).predict([r["path"] for r in rows])
        scores = {r["source"]: pr for r, pr in zip(rows, p)}

    gt = read_csv(args.gt) if args.gt else None
    results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        for frac in [float(f) for f in args.fractions.split(",")]:
            name = f"{args.strategy}_f{frac:g}_s{seed}"
            dst = Path(args.work) / name
            counts = {sp: build_set(pseudo, curated, sp, frac, args.strategy, seed, dst, scores)
                      for sp in ("train", "valid")}
            print(f"[{name}] {counts}")
            row = {"strategy": args.strategy, "fraction": frac, "seed": seed,
                   "train_counts": json.dumps(counts["train"])}
            if not args.dry_run:
                from mexp.models import Predictor
                from mexp.train import train_yolo
                best = train_yolo(dst, Path(args.work) / "runs", name, model=args.model, seed=seed)
                row["weights"] = str(best)
                if gt:
                    pred, _ = Predictor(best).predict([r["path"] for r in gt])
                    row.update({k: v for k, v in summary([r["label"] for r in gt], pred).items()})
            results.append(row)
            write_csv(out / f"runs_{args.strategy}.csv", results)

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
