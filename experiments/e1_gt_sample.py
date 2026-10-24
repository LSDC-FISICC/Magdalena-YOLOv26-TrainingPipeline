#!/usr/bin/env python3
"""E1a - Draw the independent ground-truth (GT) sample.

The GT set must not inherit labels from any colour index (otherwise comparing
the model with its own supervisor is circular), and must not contain frames
temporally adjacent to training frames. With the current random per-frame
Roboflow splits almost every frame has a training neighbour, so either sample a
new recording session (best: also covers night / another field) or retrain on
the block split of e0 and sample its held-out blocks with --block-test. The sample is drawn from the raw frames
(Google Drive / Labelbox export) or from a dataset split, stratified by the
pseudo-label of the cascade labeller and by brightness tercile so that both
classes and all illumination levels are represented. Labels are NOT written to
the manifest: annotators label blind with e1_annotate.py.

Example:
  python experiments/e1_gt_sample.py --source imagenes_labelbox --block-test \
      --n 300 --out experiments/gt/manifest.csv
  python experiments/e1_gt_sample.py --source new_session_frames --n 300
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp.data import IMG_EXT, block_split, frame_id, list_folder_dataset, unique_sources, write_csv  # noqa: E402
from mexp.pseudolabel import cascade  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="folder of raw frames, or a folder dataset split")
    ap.add_argument("--exclude-train", action="append", default=[],
                    help="dataset whose train split frames (+-gap) are excluded; repeat for every "
                         "dataset that will be trained on (all Roboflow versions)")
    ap.add_argument("--gap", type=int, default=30, help="minimum distance in frames to any train frame")
    ap.add_argument("--block-test", action="store_true",
                    help="keep only frames of the held-out test blocks of e0 --make-block-split "
                         "(use the same --block-len/--block-gap/--block-seed)")
    ap.add_argument("--block-len", type=int, default=300)
    ap.add_argument("--block-gap", type=int, default=30)
    ap.add_argument("--block-seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/gt/manifest.csv")
    args = ap.parse_args()

    src = Path(args.source)
    files = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMG_EXT]
    if any((src / c).is_dir() for c in ("plant", "no_plant")) or (src / "test").is_dir():
        files = [Path(r["path"]) for r in unique_sources(list_folder_dataset(src))]

    if args.block_test:
        assign = block_split([frame_id(p) for p in files if frame_id(p) is not None],
                             args.block_len, args.block_gap, seed=args.block_seed)
        files = [p for p in files if assign.get(frame_id(p)) == "test"]
    if args.exclude_train:
        train = {r["frame"] for d in args.exclude_train for r in list_folder_dataset(d, "train")}
        train_arr = np.array(sorted(f for f in train if f is not None))

        def far(f):
            if f is None or len(train_arr) == 0:
                return True
            i = np.searchsorted(train_arr, f)
            near = [train_arr[j] for j in (i - 1, i) if 0 <= j < len(train_arr)]
            return min(abs(f - t) for t in near) > args.gap
        files = [p for p in files if far(frame_id(p))]
    print(f"Candidate frames: {len(files)}")

    strata = {}
    for p in files:
        g = float(np.asarray(Image.open(p).convert("L")).mean())
        strata.setdefault(cascade(p), []).append((g, p))
    rng = random.Random(args.seed)
    picked = []
    per_class = args.n // max(len(strata), 1)
    for label, items in strata.items():
        items.sort()
        terciles = np.array_split(np.arange(len(items)), 3)
        for t, idx in enumerate(terciles):
            k = min(len(idx), per_class // 3 + (1 if t < per_class % 3 else 0))
            for j in rng.sample(list(idx), k):
                g, p = items[j]
                picked.append({"image_id": len(picked), "path": str(p), "frame": frame_id(p),
                               "brightness": round(g, 1), "brightness_tercile": t})
    rng.shuffle(picked)
    for i, r in enumerate(picked):
        r["image_id"] = i
    write_csv(args.out, picked)
    print(f"Wrote {len(picked)} images to {args.out} (pseudo-label strata: "
          + ", ".join(f"{k}={len(v)}" for k, v in strata.items()) + ")")


if __name__ == "__main__":
    main()
