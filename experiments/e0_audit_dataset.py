#!/usr/bin/env python3
"""E0 - Dataset audit and leakage-free re-split.

Reports, for a Roboflow folder dataset:
  * images and unique source frames per split/class (augmented copies counted once),
  * temporal leakage: test frames whose neighbour (+-k frames) is in train,
  * illumination statistics (mean grey level) to document the day/night coverage
    the paper claims,
and optionally writes a block-split copy of the dataset (symlinks) in which
train/valid/test come from disjoint temporal blocks separated by a gap.

Examples:
  python experiments/e0_audit_dataset.py --data dataset2
  python experiments/e0_audit_dataset.py --data dataset2 --make-block-split dataset2_block \
      --block-len 300 --gap 30
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp.data import list_folder_dataset, resplit, unique_sources  # noqa: E402


def brightness(path):
    return float(np.asarray(Image.open(path).convert("L")).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--neighbour", type=int, default=2, help="frames considered adjacent")
    ap.add_argument("--dark-threshold", type=float, default=50.0, help="mean grey level counted as night")
    ap.add_argument("--out", default="experiments/results/e0_audit")
    ap.add_argument("--make-block-split", metavar="OUT_DIR")
    ap.add_argument("--block-len", type=int, default=300, help="frames per block (300 = 20 s at 15 fps)")
    ap.add_argument("--gap", type=int, default=30, help="frames dropped at each block border")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list_folder_dataset(args.data)
    report = {"data": args.data, "splits": {}}
    frames = defaultdict(set)
    for split in sorted({r["split"] for r in rows}):
        sr = [r for r in rows if r["split"] == split]
        uniq = unique_sources(sr)
        frames[split] = {r["frame"] for r in uniq if r["frame"] is not None}
        report["splits"][split] = {
            "images": dict(Counter(r["label"] for r in sr)),
            "unique_frames": dict(Counter(r["label"] for r in uniq)),
        }

    tr = frames.get("train", set())
    leak = {}
    for split in ("valid", "test"):
        fs = frames.get(split, set())
        near = [f for f in fs if any((f + d) in tr for d in range(-args.neighbour, args.neighbour + 1) if d)]
        leak[split] = {"frames": len(fs), "with_train_neighbour": len(near),
                       "fraction": round(len(near) / max(len(fs), 1), 4)}
    report["temporal_leakage"] = leak

    uniq_all = unique_sources(rows)
    b = np.array([brightness(r["path"]) for r in uniq_all])
    report["illumination"] = {
        "mean_grey_percentiles_p5_p25_p50_p75_p95": np.percentile(b, [5, 25, 50, 75, 95]).round(1).tolist(),
        f"fraction_below_{args.dark_threshold:g}": round(float((b < args.dark_threshold).mean()), 4),
        "note": "Frames carry no capture time; if the night fraction is ~0 the day/night claim "
                "needs the night recordings to be added or the frames tagged with their session.",
    }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{Path(args.data).name}_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if args.make_block_split:
        dst = resplit(args.data, args.make_block_split, args.block_len, args.gap, args.seed)
        print("\nBlock split written to", dst)
        for split in ("train", "valid", "test"):
            c = Counter(r["label"] for r in list_folder_dataset(dst, split))
            print(f"  {split:<6} {dict(c)}")


if __name__ == "__main__":
    main()
