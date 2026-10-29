#!/usr/bin/env python3
"""E9 - Paired field comparison of the decision streams logged on the same frames.

During a field pass decision_logger.py records, for every camera and batch, the decision that
drove the valves (YOLO26 main engine, ``raw``), the probability of the second engine when one is
configured (``p_fallback``, e.g. EfficientNet-B0 logged with blur_threshold 0), and, with
perception:=both, the colour/NDVI index decision of each camera (/threshold<i>). It also saves
snapshots of the 320x96 crops (<camera>_<capture_ns>.png). After the snapshots are labelled blind
(e1_gt_sample.py --source <snapshot_dir>, e1_annotate.py, e1_agreement.py -> gt.csv), this script
matches each labelled snapshot with
  * the main decision of the same camera and capture stamp ("main"),
  * the second engine at --second-threshold on the same frame ("second", if logged), and
  * the first index decision of the same camera published after the capture, within
    --index-window-ms ("index", if logged; --camera-order maps cameras to /threshold<i>),
and reports accuracy, recall, specificity and MCC of each stream, and for every pair of streams
the McNemar test and the paired accuracy difference, overall and by illumination (day / night from
the capture hour, or the "condition" column of gt.csv).

Example (USB cameras of Magdalena-CV-Pipeline, EfficientNet-B0 as second engine):
  python experiments/e9_field_ab.py --gt experiments/gt_field/gt.csv --log logs/decisiones_*.csv \
      --second-name efficientnet_b0 --second-threshold 0.5 --utc-offset -6
"""

import argparse
import bisect
import glob
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp.data import read_csv, write_csv  # noqa: E402
from mexp.metrics import mcnemar, paired_accuracy_diff_ci, summary  # noqa: E402

SNAP_RE = re.compile(r"(?P<cam>.+)_(?P<ns>\d{12,})\.png$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--log", nargs="+", required=True)
    ap.add_argument("--camera-order", default="camera1,camera2,camera3,camera4,camera5",
                    help="comma list: camera i -> /threshold<i+1> (index stream only)")
    ap.add_argument("--index-window-ms", type=float, default=300)
    ap.add_argument("--main-name", default="yolo26n")
    ap.add_argument("--second-name", default="second", help="name of the engine logged as p_fallback")
    ap.add_argument("--second-threshold", type=float, default=0.5, help="p(plant) threshold for p_fallback")
    ap.add_argument("--utc-offset", type=float, default=-6, help="hours, Guatemala = -6")
    ap.add_argument("--day-hours", default="6-18", help="local hours counted as day")
    ap.add_argument("--out", default="experiments/results/e9_field_ab")
    args = ap.parse_args()

    order = args.camera_order.split(",")
    main_d, second_d, index = {}, {}, {}
    for f in (p for g in args.log for p in glob.glob(g)):
        for r in read_csv(f):
            if r.get("method", "yolo") == "yolo":
                key = (r["camera"].strip("/").split("/")[0], r["t_capture_ns"])
                main_d[key] = r["raw"] == "1"
                if r.get("p_fallback"):
                    second_d[key] = float(r["p_fallback"]) >= args.second_threshold
            else:
                index.setdefault(int(r["index"]), []).append((int(r["t_logger_rx_ns"]), r["raw"] == "1"))
    for v in index.values():
        v.sort()
    d0, d1 = (int(h) for h in args.day_hours.split("-"))
    names = {"main": args.main_name, "second": args.second_name, "index": "index"}

    rows = []
    for g in read_csv(args.gt):
        m = SNAP_RE.search(Path(g["path"]).name)
        if not m:
            continue
        cam, ns = m["cam"], m["ns"]
        dec = {}
        if (cam, ns) in main_d:
            dec["main"] = main_d[(cam, ns)]
        if (cam, ns) in second_d:
            dec["second"] = second_d[(cam, ns)]
        if cam in order and order.index(cam) in index:
            seq = index[order.index(cam)]
            j = bisect.bisect_left(seq, (int(ns), False))
            if j < len(seq) and seq[j][0] - int(ns) <= args.index_window_ms * 1e6:
                dec["index"] = seq[j][1]
        if not dec:
            continue
        local = datetime.fromtimestamp(int(ns) / 1e9, timezone.utc) + timedelta(hours=args.utc_offset)
        cond = g.get("condition") or ("day" if d0 <= local.hour < d1 else "night")
        rows.append({"path": g["path"], "camera": cam, "label": g["label"], "condition": cond,
                     **{names[k]: ("plant" if v else "no_plant") for k, v in dec.items()}})
    if not rows:
        sys.exit("No labelled snapshot matched a logged decision (check --log and the snapshot names)")

    streams = [names[k] for k in ("main", "second", "index") if any(names[k] in r for r in rows)]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_predictions.csv", rows, ["path", "camera", "label", "condition", *streams])
    report = {}
    for cond in ["all"] + sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if cond == "all" or r["condition"] == cond]
        rep = {"n": len(sub), "streams": {}, "pairs": {}}
        for s in streams:
            ss = [r for r in sub if s in r]
            if ss:
                rep["streams"][s] = summary([r["label"] for r in ss], [r[s] for r in ss])
                print(f"[{cond}] {s:<16} n={len(ss):<5} acc {100 * rep['streams'][s]['accuracy']:.1f}%  "
                      f"recall {rep['streams'][s]['recall']:.2f}  specificity {rep['streams'][s]['specificity']:.2f}")
        for a, b in [(x, y) for i, x in enumerate(streams) for y in streams[i + 1:]]:
            ss = [r for r in sub if a in r and b in r]
            if not ss:
                continue
            yt = [r["label"] for r in ss]
            only_a, only_b, p = mcnemar(yt, [r[a] for r in ss], [r[b] for r in ss])
            lo, hi = paired_accuracy_diff_ci(yt, [r[a] for r in ss], [r[b] for r in ss])
            rep["pairs"][f"{a} vs {b}"] = {"n": len(ss), "diff_pp_ci": [lo, hi], f"only_{a}_correct": only_a,
                                           f"only_{b}_correct": only_b, "mcnemar_p": p}
            print(f"[{cond}] {a} vs {b}: n={len(ss)}  diff CI [{lo:+.1f}, {hi:+.1f}] pp  "
                  f"only {a} {only_a} / only {b} {only_b}  McNemar p={p:.2g}")
        report[cond] = rep
    (out / "field_ab.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
