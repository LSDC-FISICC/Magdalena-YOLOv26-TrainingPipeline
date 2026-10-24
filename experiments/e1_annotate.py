#!/usr/bin/env python3
"""E1b - Blind manual annotation of the GT manifest (one CSV per annotator).

Keys:  p = plant   n = no_plant   u = unsure (excluded from GT)   b = back   q = quit
Illumination is not asked here: add a "condition" column (day / dusk / night)
to the manifest from the recording session so E2 can report per-condition results.

The protocol: two annotators label independently, then e1_agreement.py computes
Cohen's kappa and lists the disagreements for adjudication. "plant" means any
living sugarcane tissue inside the crop window that should receive vinasse.

Example:
  python experiments/e1_annotate.py --manifest experiments/gt/manifest.csv --annotator A
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp.data import read_csv, write_csv  # noqa: E402

KEYS = {ord("p"): "plant", ord("n"): "no_plant", ord("u"): "unsure"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--scale", type=float, default=3.0, help="display zoom")
    args = ap.parse_args()

    rows = read_csv(args.manifest)
    out = Path(args.manifest).with_name(f"labels_{args.annotator}.csv")
    labels = {r["image_id"]: r["label"] for r in read_csv(out)} if out.exists() else {}
    i = next((k for k, r in enumerate(rows) if r["image_id"] not in labels), len(rows))
    while 0 <= i < len(rows):
        r = rows[i]
        img = cv2.imread(r["path"])
        img = cv2.resize(img, (320, 96))  # what the model sees
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"{i + 1}/{len(rows)}  [{labels.get(r['image_id'], '-')}]  p/n/u b q",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(f"GT annotator {args.annotator}", img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        if k == ord("b"):
            i -= 1
            continue
        if k in KEYS:
            labels[r["image_id"]] = KEYS[k]
            write_csv(out, [{"image_id": k_, "label": v} for k_, v in labels.items()], ["image_id", "label"])
            i += 1
    cv2.destroyAllWindows()
    print(f"{len(labels)}/{len(rows)} labelled -> {out}")


if __name__ == "__main__":
    main()
