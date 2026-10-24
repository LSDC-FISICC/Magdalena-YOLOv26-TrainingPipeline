#!/usr/bin/env python3
"""E9 - Paired field comparison: YOLO26 vs. the colour/NDVI index pipeline on the same frames.

During a field pass both pipelines run on the Jetson (one drives the valves,
the other is only logged) and decision_logger.py stores both decision streams
plus snapshots of the 320x96 crops (<camera>_<capture_ns>.png). After the
snapshots are labelled blind (e1_gt_sample.py --source <snapshot_dir>,
e1_annotate.py, e1_agreement.py -> gt.csv), this script matches each labelled
snapshot with
  * the YOLO decision of the same camera and capture stamp, and
  * the first index decision of the same camera published after the capture
    (within --index-window-ms), using --camera-order to map cameras to /threshold<i>,
and reports accuracy, recall, specificity, MCC and the McNemar test of the
two pipelines, overall and by illumination (day / night from the capture hour,
or the "condition" column of gt.csv). This is the comparison with the system
actually deployed today (pseudo-NDVI), which the offline E2 cannot reproduce
because the training frames carry no infrared channel.

Example:
  python experiments/e9_field_ab.py --gt experiments/gt_field/gt.csv --log logs/decisions_*.csv \
      --camera-order camera_55,camera_56,camera_57,camera_58,camera_59 --utc-offset -6
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
    ap.add_argument("--camera-order", required=True, help="comma list: camera i -> /threshold<i+1>")
    ap.add_argument("--index-window-ms", type=float, default=300)
    ap.add_argument("--utc-offset", type=float, default=-6, help="hours, Guatemala = -6")
    ap.add_argument("--day-hours", default="6-18", help="local hours counted as day")
    ap.add_argument("--out", default="experiments/results/e9_field_ab")
    args = ap.parse_args()

    order = args.camera_order.split(",")
    yolo, index = {}, {}
    for f in (p for g in args.log for p in glob.glob(g)):
        for r in read_csv(f):
            if r["method"] == "yolo":
                yolo[(r["camera"].strip("/").split("/")[0], r["t_capture_ns"])] = r["raw"] == "1"
            else:
                index.setdefault(int(r["index"]), []).append((int(r["t_logger_rx_ns"]), r["raw"] == "1"))
    for v in index.values():
        v.sort()
    d0, d1 = (int(h) for h in args.day_hours.split("-"))

    rows = []
    for g in read_csv(args.gt):
        m = SNAP_RE.search(Path(g["path"]).name)
        if not m:
            continue
        cam, ns = m["cam"], m["ns"]
        y = yolo.get((cam, ns))
        x = None
        if cam in order and order.index(cam) in index:
            seq = index[order.index(cam)]
            j = bisect.bisect_left(seq, (int(ns), False))
            if j < len(seq) and seq[j][0] - int(ns) <= args.index_window_ms * 1e6:
                x = seq[j][1]
        if y is None or x is None:
            continue
        local = datetime.fromtimestamp(int(ns) / 1e9, timezone.utc) + timedelta(hours=args.utc_offset)
        cond = g.get("condition") or ("day" if d0 <= local.hour < d1 else "night")
        rows.append({"path": g["path"], "label": g["label"], "condition": cond,
                     "yolo": "plant" if y else "no_plant", "index": "plant" if x else "no_plant"})
    if not rows:
        sys.exit("No labelled snapshot matched both decision streams")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "paired_predictions.csv", rows)
    report = {}
    for cond in ["all"] + sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if cond == "all" or r["condition"] == cond]
        yt = [r["label"] for r in sub]
        b, c, p = mcnemar(yt, [r["yolo"] for r in sub], [r["index"] for r in sub])
        lo, hi = paired_accuracy_diff_ci(yt, [r["yolo"] for r in sub], [r["index"] for r in sub])
        report[cond] = {"yolo": summary(yt, [r["yolo"] for r in sub]),
                        "index": summary(yt, [r["index"] for r in sub]),
                        "diff_pp_ci": [lo, hi], "only_yolo_correct": b, "only_index_correct": c, "mcnemar_p": p}
        print(f"[{cond}] n={len(sub)}  YOLO {100 * report[cond]['yolo']['accuracy']:.1f}%  "
              f"index {100 * report[cond]['index']['accuracy']:.1f}%  diff CI [{lo:+.1f}, {hi:+.1f}] pp  p={p:.2g}")
    (out / "field_ab.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
