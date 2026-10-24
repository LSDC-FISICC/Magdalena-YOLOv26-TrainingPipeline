#!/usr/bin/env python3
"""E0b - Agreement between the labels of different pseudo-labelling variants.

All Roboflow projects are built from the same video frames (same frame_XXXXX
names), so the label each variant gives to a shared frame can be compared
directly. For every pair of datasets the script reports the number of shared
source frames, the observed agreement, Cohen's kappa and the direction of the
disagreements; with --reference it also reports each variant against one
dataset (e.g. the curated one). This quantifies how noisy each colour-index
labeller is and documents the ground-truth contamination argument.

Example:
  python experiments/e0_label_agreement.py --reference sugarcane_exg_hsv_lab_v3-2-100 \
      --dataset roboflow/*
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp.data import list_folder_dataset, unique_sources, write_csv  # noqa: E402
from mexp.metrics import cohen_kappa  # noqa: E402


def labels(root):
    return {r["source"]: r["label"] for r in unique_sources(list_folder_dataset(root))}


def compare(a, b):
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"shared": 0}
    ka, po = cohen_kappa([a[s] for s in shared], [b[s] for s in shared])
    return {"shared": len(shared), "agreement": round(po, 4), "kappa": round(ka, 4),
            "a_plant_b_no": sum(a[s] == "plant" and b[s] == "no_plant" for s in shared),
            "a_no_b_plant": sum(a[s] == "no_plant" and b[s] == "plant" for s in shared)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--reference", help="name (directory basename) of the reference dataset")
    ap.add_argument("--out", default="experiments/results/e0_label_agreement")
    args = ap.parse_args()

    sets = {Path(d).name: labels(d) for d in args.dataset if Path(d).is_dir()}
    for k, v in sets.items():
        print(f"{k:<34} frames={len(v):>5}  plant={sum(x == 'plant' for x in v.values()):>5}")
    pairs = []
    for x, y in itertools.combinations(sorted(sets), 2):
        pairs.append({"a": x, "b": y, **compare(sets[x], sets[y])})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pairwise.csv", pairs, list(dict.fromkeys(k for p in pairs for k in p)))
    if args.reference:
        ref = sets[args.reference]
        rows = [{"dataset": k, **compare(v, ref)} for k, v in sorted(sets.items()) if k != args.reference]
        write_csv(out / f"vs_{args.reference}.csv", rows, list(dict.fromkeys(k for r in rows for k in r)))
        print(f"\nAgainst {args.reference} (a = variant, b = reference):")
        for r in rows:
            if r["shared"]:
                print(f"  {r['dataset']:<34} shared={r['shared']:>5}  agree={100 * r['agreement']:5.1f}%  "
                      f"kappa={r['kappa']:.3f}  plant->no={r['a_plant_b_no']:>4}  no->plant={r['a_no_b_plant']:>4}")
        (out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
