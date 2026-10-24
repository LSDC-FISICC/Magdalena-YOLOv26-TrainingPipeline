"""Dataset listing, source-frame parsing and leakage-free splitting."""

import csv
import os
import re
from pathlib import Path

from . import CLASSES

IMG_EXT = {".jpg", ".jpeg", ".png"}
# Roboflow renames "frame_01232.png" to "frame_01232_png.rf.<hash>.jpg" and keeps
# one file per augmented copy, so the source frame is the part before "_png.rf."
_FRAME_RE = re.compile(r"frame_(\d+)")


def frame_id(path):
    """Integer index of the source video frame, or None if the name has no frame number."""
    m = _FRAME_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


def source_key(path):
    """Name of the original (pre-Roboflow) image, shared by all augmented copies."""
    name = Path(path).name
    return name.split("_png.rf.")[0].split(".rf.")[0]


def list_folder_dataset(root, split=None):
    """Return [{path, label, split, frame, source}] for a folder-structured classification set.

    ``root`` may be the dataset root (with train/valid/test) or a single split folder.
    """
    root = Path(root)
    splits = [split] if split else [d.name for d in sorted(root.iterdir()) if d.is_dir()]
    has_splits = all((root / s).is_dir() and any((root / s / c).is_dir() for c in CLASSES) for s in splits)
    if not has_splits:
        splits = [None]
    rows = []
    for s in splits:
        base = root / s if s else root
        for c in CLASSES:
            d = base / c
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    rows.append({"path": str(f), "label": c, "split": s or "",
                                 "frame": frame_id(f), "source": source_key(f)})
    return rows


def unique_sources(rows):
    """Keep one image per source frame (drops Roboflow augmented duplicates)."""
    seen, out = set(), []
    for r in rows:
        if r["source"] not in seen:
            seen.add(r["source"])
            out.append(r)
    return out


def block_of(frame, block_len=300, seed=0, ratios=(0.7, 0.15, 0.15)):
    """Split of the temporal block containing ``frame``.

    The assignment depends only on (block index, seed), so every Roboflow
    version built from the same video gets the same held-out blocks.
    """
    import random
    u = random.Random(f"{seed}-{frame // block_len}").random()
    return "train" if u < ratios[0] else "valid" if u < ratios[0] + ratios[1] else "test"


def block_split(frames, block_len=300, gap=30, ratios=(0.7, 0.15, 0.15), seed=0):
    """Assign frame indices to train/valid/test by contiguous temporal blocks.

    Consecutive frames of a tractor video are nearly identical, so a random
    per-frame split puts neighbours of every test frame in train. Blocks of
    ``block_len`` frames are assigned to a split as a whole, and the ``gap``
    frames next to a block of a different split are dropped so no split
    touches another. Returns {frame: split or None(dropped)}.
    """
    assign = {}
    for f in set(frames):
        lo = (f // block_len) * block_len
        split = block_of(f, block_len, seed, ratios)
        prev_differs = block_of(lo - 1, block_len, seed, ratios) != split if lo else False
        next_differs = block_of(lo + block_len, block_len, seed, ratios) != split
        border = ((f - lo) < gap and prev_differs) or ((lo + block_len - 1 - f) < gap and next_differs)
        assign[f] = None if border else split
    return assign


def resplit(src, dst, block_len=300, gap=30, seed=0):
    """Write (symlinks) a copy of folder dataset ``src`` re-split by temporal blocks.

    Augmented copies stay only in train; valid/test keep one image per source frame.
    """
    rows = list_folder_dataset(src)
    assign = block_split([r["frame"] for r in rows if r["frame"] is not None], block_len, gap, seed=seed)
    seen = set()
    for r in rows:
        split = assign.get(r["frame"])
        if split is None:
            continue
        if split != "train":
            if (split, r["source"]) in seen:
                continue
            seen.add((split, r["source"]))
        link_or_copy(r["path"], Path(dst) / split / r["label"] / Path(r["path"]).name)
    return dst


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def link_or_copy(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        import shutil
        shutil.copy(src, dst)
