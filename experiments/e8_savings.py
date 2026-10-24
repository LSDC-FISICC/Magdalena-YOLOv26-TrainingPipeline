#!/usr/bin/env python3
"""E8 - Estimated vinasse saving and valve-policy trade-off (paper: "[XX%] reduction in applied volume").

Decision sequences come either from the field (decision_logger.py CSV logs: one
row per camera per batch) or offline, by running a model over an ordered
sequence of frames (--frames: a folder or folder dataset; frames sorted by
number and timed at --fps, so it must be a contiguous recording, not a random
split). For every valve policy (see mexp/debounce.py; "time:500" is the rule
deployed in modbus_controller) the script reports:
  * fraction of time the valves are closed -> volume reduction vs. always open,
  * valve switches per valve-minute (actuator wear / oscillation),
  * with per-frame labels: accuracy of the valve state, under-application
    (valve closed over cane, % of cane frames) and residual waste (open over
    soil, % of soil frames), plus the ideal reduction (soil fraction),
  * with --gt-metrics (e2 metrics.json of the deployed model): the observed
    closed fraction corrected for classifier error (Rogan-Gladen) = estimated
    true gap fraction of the field.

The % reduction does not depend on flow or speed; litres per hectare need
--flow-lpm (one valve), --speed-kmh and --furrow-spacing-m. Distances use --speed-kmh.

Examples:
  python experiments/e8_savings.py --log logs/decisions_*.csv \
      --policies none,time:500,kofn:3/2,asym:3/2 --speed-kmh 6 \
      --gt-metrics experiments/results/e2_gt/metrics.json --model-name yolo26s_curated
  python experiments/e8_savings.py --frames recording_session_frames/ --weights best.pt
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp import debounce  # noqa: E402
from mexp.data import list_folder_dataset, read_csv, unique_sources, write_csv  # noqa: E402


def sequences_from_logs(paths, method="yolo"):
    """{(log, camera): [(t_seconds, raw_plant, label=None)]} for one perception method."""
    seq = {}
    for f in paths:
        for r in read_csv(f):
            if r.get("method", "yolo") != method:
                continue
            t = int(r["t_capture_ns"] or r["t_logger_rx_ns"]) / 1e9
            seq.setdefault((Path(f).stem, r["camera"]), []).append((t, r["raw"] == "1", None))
    return {k: sorted(v) for k, v in seq.items()}


def sequences_from_frames(folder, weights, fps):
    from mexp.models import Predictor
    rows = sorted((r for r in unique_sources(list_folder_dataset(folder)) if r["frame"] is not None),
                  key=lambda r: r["frame"])
    pred, _ = Predictor(weights).predict([r["path"] for r in rows])
    labelled = all(r["label"] for r in rows)
    seq = [(r["frame"] / fps, p == "plant", (r["label"] == "plant") if labelled else None)
           for r, p in zip(rows, pred)]
    return {("offline", Path(folder).name): seq}


def rogan_gladen(observed_plant, sens, spec):
    den = sens + spec - 1
    return float(np.clip((observed_plant + spec - 1) / den, 0, 1)) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--log", nargs="+")
    g.add_argument("--frames")
    ap.add_argument("--weights")
    ap.add_argument("--method", default="yolo", help="decisions to use from the logs: yolo or index")
    ap.add_argument("--policies", default="none,time:500,kofn:3/2,kofn:5/3,asym:3/2")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--valves-per-furrow", type=int, default=4)
    ap.add_argument("--flow-lpm", type=float, default=float("nan"), help="flow of one valve (L/min)")
    ap.add_argument("--speed-kmh", type=float, default=float("nan"))
    ap.add_argument("--furrow-spacing-m", type=float, default=1.5)
    ap.add_argument("--gt-metrics", help="metrics.json from e2_eval_gt.py")
    ap.add_argument("--model-name", help="method name inside --gt-metrics")
    ap.add_argument("--out", default="experiments/results/e8_savings")
    args = ap.parse_args()

    if args.log:
        seqs = sequences_from_logs([p for f in args.log for p in glob.glob(f)], args.method)
    else:
        if not args.weights:
            sys.exit("--frames needs --weights")
        seqs = sequences_from_frames(args.frames, args.weights, args.fps)

    gtm = None
    if args.gt_metrics:
        m = [r for r in json.loads(Path(args.gt_metrics).read_text())
             if r["method"] == args.model_name and r["condition"] == "all"]
        gtm = m[0] if m else None
    speed_cms = args.speed_kmh / 3.6 * 100

    rows = []
    for policy in args.policies.split(","):
        n_dec = n_closed = n_toggles = 0
        valve_s = 0.0
        lab = {"n": 0, "ok": 0, "plant": 0, "soil": 0, "closed_on_plant": 0, "open_on_soil": 0}
        for seq in seqs.values():
            times = [s[0] for s in seq]
            states = debounce.run([s[1] for s in seq], times, policy)
            n_dec += len(states)
            n_closed += sum(1 for s in states if not s)
            n_toggles += debounce.toggles(states)
            valve_s += (times[-1] - times[0]) if len(times) > 1 else 0
            for (_, _, y), st in zip(seq, states):
                if y is None:
                    continue
                lab["n"] += 1; lab["ok"] += int(st == y)
                lab["plant"] += int(y); lab["soil"] += int(not y)
                lab["closed_on_plant"] += int(y and not st); lab["open_on_soil"] += int(not y and st)
        closed = n_closed / max(n_dec, 1)
        row = {"policy": policy, "decisions": n_dec, "closed_fraction": closed,
               "volume_reduction_pct": 100 * closed,
               "switches_per_valve_minute": n_toggles / (valve_s / 60) if valve_s else float("nan")}
        if policy.startswith("time:") and speed_cms == speed_cms:
            row["blocked_distance_after_switch_cm"] = speed_cms * float(policy[5:]) / 1000
        if lab["n"]:
            row.update({"state_accuracy": lab["ok"] / lab["n"],
                        "under_application_pct_of_cane": 100 * lab["closed_on_plant"] / max(lab["plant"], 1),
                        "waste_pct_of_soil": 100 * lab["open_on_soil"] / max(lab["soil"], 1),
                        "ideal_reduction_pct": 100 * lab["soil"] / lab["n"]})
        if gtm:
            row.update({"est_true_gap_pct": 100 * (1 - rogan_gladen(1 - closed, gtm["recall"], gtm["specificity"])),
                        "classifier_under_application_pct_of_cane": 100 * (1 - gtm["recall"])})
        if args.flow_lpm == args.flow_lpm and speed_cms == speed_cms:
            area_rate_ha_min = args.speed_kmh * 1000 / 60 * args.furrow_spacing_m / 1e4
            dose = args.valves_per_furrow * args.flow_lpm / area_rate_ha_min
            row.update({"dose_open_l_per_ha": dose, "saved_l_per_ha": dose * closed})
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "savings.csv", rows, list(dict.fromkeys(k for r in rows for k in r)))


if __name__ == "__main__":
    main()
