#!/usr/bin/env python3
"""E10 - Collect the experiment outputs into LaTeX macros for the manuscript placeholders.

Writes a results_macros.tex defining one macro per number of the abstract and
introduction; a macro whose experiment has not been run yet expands to the red
\\todoNote{...} of the manuscript, so the paper always compiles and shows what
is missing. \\input{results_macros.tex} after the \\todoNote definition.

  \\ResNGT, \\ResKappa            E1  agreement.json
  \\ResAccGT, \\ResDeltaPP        E2  metrics.json / pairwise.csv (model vs supervising index)
  \\ResCurationPct, \\ResCurationGrid   E3 summary_<strategy>.csv
  \\ResThroughput, \\ResInferMs   E6  <tag>.json (batch 5)
  \\ResLatencyMs, \\ResFPS        E7  latency.json
  \\ResVolumeReduction           E8  savings.csv (deployed policy)
  \\ResFieldAccYolo, \\ResFieldAccIndex  E9 field_ab.json

Example:
  python experiments/e10_paper_macros.py --model yolo26s_curated --supervisor cascade_hsv_lab_exgr \
      --e6-tag yolo26s_fp16 --policy time:500 --out paper/results_macros.tex
"""

import argparse
import csv
import json
from pathlib import Path

R = Path("experiments/results")


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def load_csv(p):
    p = Path(p)
    return list(csv.DictReader(open(p))) if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(R))
    ap.add_argument("--gt-dir", default="experiments/gt")
    ap.add_argument("--e2-dir", default="e2_gt")
    ap.add_argument("--model", required=True, help="method name of the deployed model in E2")
    ap.add_argument("--supervisor", default="cascade_hsv_lab_exgr")
    ap.add_argument("--e3-strategy", default="random")
    ap.add_argument("--e3-tolerance-pp", type=float, default=1.0,
                    help="smallest curated fraction whose mean accuracy is within this of 100%% curation")
    ap.add_argument("--e6-tag", default="yolo26s_fp16")
    ap.add_argument("--policy", default="time:500")
    ap.add_argument("--out", default="experiments/results/results_macros.tex")
    args = ap.parse_args()
    res = Path(args.results)
    m = {}

    agr = load_json(Path(args.gt_dir) / "agreement.json")
    if agr:
        m["ResNGT"] = f"{agr['n_gt']}"
        m["ResKappa"] = f"{agr['cohen_kappa']:.2f}"

    met = load_json(res / args.e2_dir / "metrics.json")
    if met:
        acc = {r["method"]: r for r in met if r["condition"] == "all"}
        if args.model in acc:
            r = acc[args.model]
            m["ResAccGT"] = f"{100 * r['accuracy']:.1f}\\,\\%"
            m["ResAccGTCI"] = f"{100 * r['acc_ci_low']:.1f}--{100 * r['acc_ci_high']:.1f}\\,\\%"
    pw = load_csv(res / args.e2_dir / "pairwise.csv")
    if pw:
        for r in pw:
            if r["model"] == args.model and r["baseline"] == args.supervisor:
                m["ResDeltaPP"] = f"{float(r['diff_pp']):.1f}"
                m["ResDeltaPPCI"] = f"{float(r['diff_ci_low_pp']):.1f}--{float(r['diff_ci_high_pp']):.1f}"

    cur = load_csv(res / "e3_curation" / f"summary_{args.e3_strategy}.csv")
    if cur:
        cur = sorted(cur, key=lambda r: float(r["fraction"]))
        full = float(cur[-1]["acc_mean"])
        ok = [r for r in cur if 100 * (full - float(r["acc_mean"])) <= args.e3_tolerance_pp]
        m["ResCurationPct"] = f"{float(ok[0]['fraction']):g}\\,\\%"
        m["ResCurationGrid"] = "/".join(f"{float(r['fraction']):g}" for r in cur) + "\\,\\%"

    e6 = load_json(res / "e6_latency" / f"{args.e6_tag}.json")
    if e6:
        b5 = [r for r in e6["results"] if r["batch"] == 5] or e6["results"][-1:]
        m["ResThroughput"] = f"{b5[0]['throughput_img_s']:.0f}"
        m["ResInferMs"] = f"{b5[0]['mean_ms']:.2f}\\,ms"

    e7 = load_json(res / "e7_latency" / "latency.json")
    if e7:
        t = e7["total"]["to_open_ms"]
        if t == t:
            m["ResLatencyMs"] = f"{t:.0f}\\,ms"
        if e7.get("batch_rate_hz"):
            m["ResFPS"] = f"{e7['batch_rate_hz']:.0f}"

    sav = load_csv(res / "e8_savings" / "savings.csv")
    if sav:
        r = [x for x in sav if x.get("policy") == args.policy] or sav[:1]
        v = r[0].get("est_true_gap_pct") or r[0]["volume_reduction_pct"]
        m["ResVolumeReduction"] = f"{float(v):.0f}\\,\\%"

    ab = load_json(res / "e9_field_ab" / "field_ab.json")
    if ab:
        m["ResFieldAccYolo"] = f"{100 * ab['all']['yolo']['accuracy']:.1f}\\,\\%"
        m["ResFieldAccIndex"] = f"{100 * ab['all']['index']['accuracy']:.1f}\\,\\%"

    pending = {"ResNGT": "$n$", "ResKappa": "\\ldots", "ResAccGT": "XX\\,\\%", "ResAccGTCI": "XX--XX\\,\\%",
               "ResDeltaPP": "XX", "ResDeltaPPCI": "XX--XX", "ResCurationPct": "XX\\,\\%",
               "ResCurationGrid": "0/10/50/100\\,\\%", "ResThroughput": "XX", "ResInferMs": "XX\\,ms",
               "ResLatencyMs": "XX\\,ms", "ResFPS": "15", "ResVolumeReduction": "XX\\,\\%",
               "ResFieldAccYolo": "XX\\,\\%", "ResFieldAccIndex": "XX\\,\\%"}
    lines = ["% Generated by experiments/e10_paper_macros.py -- do not edit by hand"]
    for k, default in pending.items():
        value = m.get(k, "\\todoNote{" + default + "}")
        lines.append("\\newcommand{\\" + k + "}{" + value + "}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nfilled {len(m)}/{len(pending)} -> {args.out}")


if __name__ == "__main__":
    main()
