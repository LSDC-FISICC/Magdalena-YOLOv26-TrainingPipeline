#!/usr/bin/env python3
"""E2 - Classifier vs. the colour indices that supervised it, on the independent GT.

Fills the abstract placeholders "accuracy [XX%]" and "exceeded the colour index
that supervised it by [XX] percentage points". Every method is run on exactly
the same images; the output contains:
  * predictions.csv      per-image prediction and p(plant) of every method,
  * metrics.csv / .json  accuracy (Wilson 95% CI), precision, recall, specificity,
                         F1, balanced accuracy, MCC, overall and per condition,
  * pairwise.csv         McNemar exact test and bootstrap CI of the accuracy
                         difference (pp) of each model against each baseline,
  * table.tex            LaTeX table for the paper (plain hline rules, no extra packages).

Example:
  python experiments/e2_eval_gt.py --gt experiments/gt/gt.csv \
      --model yolo26n_curated=runs/classify/magdalena/plant_cls_v4/weights/best.pt \
      --model yolo26n_uncurated=runs/classify/magdalena/plant_cls_v3/weights/best.pt
  # a folder dataset split can be used instead of gt.csv (e.g. a pseudo-labelled test):
  python experiments/e2_eval_gt.py --gt-folder dataset2/test --model v4=...best.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp.data import list_folder_dataset, read_csv, write_csv  # noqa: E402
from mexp.metrics import mcnemar, paired_accuracy_diff_ci, summary  # noqa: E402
from mexp.pseudolabel import LABELERS  # noqa: E402


def latex_table(rows, caption):
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
             r"\begin{tabular}{lcccccc}", r"\hline",
             r"Method & Acc. (\%) & 95\% CI & Prec. & Rec. & Spec. & MCC \\", r"\hline"]
    for r in rows:
        lines.append(f"{r['method'].replace('_', ' ')} & {100 * r['accuracy']:.1f} & "
                     f"[{100 * r['acc_ci_low']:.1f}, {100 * r['acc_ci_high']:.1f}] & {r['precision']:.3f} & "
                     f"{r['recall']:.3f} & {r['specificity']:.3f} & {r['mcc']:.3f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gt", help="CSV with path,label[,condition]")
    g.add_argument("--gt-folder", help="folder with plant/ and no_plant/")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=WEIGHTS",
                    help=".pt / .engine / .onnx (YOLO) or .pth (torchvision); repeatable")
    ap.add_argument("--baselines", default=",".join(LABELERS), help="comma list of colour labellers")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5, help="p(plant) decision threshold")
    ap.add_argument("--out", default="experiments/results/e2_gt")
    args = ap.parse_args()

    if args.gt:
        gt = read_csv(args.gt)
    else:
        gt = list_folder_dataset(args.gt_folder)
    paths = [r["path"] for r in gt]
    y = [r["label"] for r in gt]
    cond = [r.get("condition") or "all" for r in gt]
    print(f"GT images: {len(gt)}  (plant={y.count('plant')}, no_plant={y.count('no_plant')})")

    preds, probs = {}, {}
    baselines = [b for b in args.baselines.split(",") if b]
    for b in baselines:
        preds[b] = [LABELERS[b](p) for p in paths]
    models = []
    for spec in args.model:
        name, w = spec.split("=", 1)
        from mexp.models import Predictor
        preds[name], probs[name] = Predictor(w, batch=args.batch).predict(paths, args.threshold)
        models.append(name)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "predictions.csv", [
        {"path": p, "label": t, "condition": c, **{m: preds[m][i] for m in preds},
         **{f"p_{m}": round(probs[m][i], 5) for m in probs}}
        for i, (p, t, c) in enumerate(zip(paths, y, cond))])

    metrics = []
    for m in preds:
        for c in ["all"] + sorted(set(cond) - {"all"}):
            idx = [i for i in range(len(y)) if c == "all" or cond[i] == c]
            s = summary([y[i] for i in idx], [preds[m][i] for i in idx])
            metrics.append({"method": m, "condition": c, **s})
    write_csv(out / "metrics.csv", metrics)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pairwise = []
    for m in models:
        for b in baselines:
            bb, cc, p = mcnemar(y, preds[m], preds[b])
            lo, hi = paired_accuracy_diff_ci(y, preds[m], preds[b])
            acc_m = summary(y, preds[m])["accuracy"]; acc_b = summary(y, preds[b])["accuracy"]
            pairwise.append({"model": m, "baseline": b, "diff_pp": round(100 * (acc_m - acc_b), 2),
                             "diff_ci_low_pp": round(lo, 2), "diff_ci_high_pp": round(hi, 2),
                             "only_model_correct": bb, "only_baseline_correct": cc, "mcnemar_p": p})
    if pairwise:
        write_csv(out / "pairwise.csv", pairwise)

    overall = [r for r in metrics if r["condition"] == "all"]
    (out / "table.tex").write_text(latex_table(
        overall, f"Classification on the independently annotated set ($n={len(y)}$)."))
    print(f"\n{'method':<28}{'acc':>7}{'95% CI':>18}{'rec':>7}{'spec':>7}{'mcc':>7}")
    for r in overall:
        print(f"{r['method']:<28}{100 * r['accuracy']:>6.1f}%  [{100 * r['acc_ci_low']:.1f}, "
              f"{100 * r['acc_ci_high']:.1f}]{r['recall']:>7.3f}{r['specificity']:>7.3f}{r['mcc']:>7.3f}")
    for r in pairwise:
        print(f"{r['model']} vs {r['baseline']}: {r['diff_pp']:+.1f} pp "
              f"[{r['diff_ci_low_pp']:+.1f}, {r['diff_ci_high_pp']:+.1f}]  McNemar p={r['mcnemar_p']:.2g}")
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
