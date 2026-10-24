#!/usr/bin/env python3
"""E1c - Inter-annotator agreement and consensus GT.

Computes Cohen's kappa between two annotators (the "[n images, two annotators,
Cohen's kappa = ...]" placeholder of the paper), writes the disagreements for
adjudication and, once an adjudication CSV exists, the final GT
(experiments/gt/gt.csv: image_id, path, label, + manifest columns).

Example:
  python experiments/e1_agreement.py --manifest experiments/gt/manifest.csv \
      --a experiments/gt/labels_A.csv --b experiments/gt/labels_B.csv \
      [--adjudication experiments/gt/adjudication.csv]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp.data import read_csv, write_csv  # noqa: E402
from mexp.metrics import cohen_kappa  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--adjudication", help="CSV image_id,label resolving the disagreements")
    args = ap.parse_args()

    man = {r["image_id"]: r for r in read_csv(args.manifest)}
    la = {r["image_id"]: r["label"] for r in read_csv(args.a)}
    lb = {r["image_id"]: r["label"] for r in read_csv(args.b)}
    both = [i for i in man if i in la and i in lb and "unsure" not in (la[i], lb[i])]
    kappa, po = cohen_kappa([la[i] for i in both], [lb[i] for i in both])
    disagree = [i for i in both if la[i] != lb[i]]
    unsure = [i for i in man if "unsure" in (la.get(i), lb.get(i))]
    gt_dir = Path(args.manifest).parent
    write_csv(gt_dir / "disagreements.csv",
              [{"image_id": i, "path": man[i]["path"], "A": la[i], "B": lb[i], "label": ""} for i in disagree]
              or [{"image_id": "", "path": "", "A": "", "B": "", "label": ""}])

    adj = {r["image_id"]: r["label"] for r in read_csv(args.adjudication)} if args.adjudication else {}
    gt = []
    for i in both:
        label = la[i] if la[i] == lb[i] else adj.get(i)
        if label in ("plant", "no_plant"):
            gt.append({**man[i], "label": label})
    report = {"n_manifest": len(man), "n_both_labelled": len(both), "n_unsure_excluded": len(unsure),
              "observed_agreement": po, "cohen_kappa": kappa, "n_disagreements": len(disagree),
              "n_adjudicated": sum(1 for i in disagree if i in adj), "n_gt": len(gt),
              "gt_plant": sum(r["label"] == "plant" for r in gt)}
    if gt:
        write_csv(gt_dir / "gt.csv", gt)
    (gt_dir / "agreement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if len(disagree) > len(adj):
        print(f"\nAdjudicate {gt_dir / 'disagreements.csv'} (fill 'label') and pass it as --adjudication.")


if __name__ == "__main__":
    main()
