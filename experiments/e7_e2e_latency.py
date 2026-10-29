#!/usr/bin/env python3
"""E7 - Decomposed camera-to-valve latency (paper: "camera-to-valve latency [XX ms]").

Input: one or more CSV logs written by ros2_ws/tools/decision_logger.py during a
run of the full graph (Magdalena-Camera -> cameraNode -> inference_node ->
modbus_controller -> PLC -> valves; the logger listens to the same topic as
modbus_controller). All stamps come from the
Jetson clock, so the cameras must be driven from the same host (or PTP/chrony
synchronised). Stages (ms):

  acquisition+buffer+preproc  t_preprocessed - t_capture   (includes waiting for the 15 Hz timer)
  transport to inference      t_received - t_preprocessed  (DDS of the 5x320x96 float batch)
  inference                   t_infer_done - t_received    (split + TensorRT + copy back)
  transport to controller     t_logger_rx - t_infer_done (same topic as modbus_controller)
  Modbus write                measured with --modbus-host (one write_single_coil per message)
  valve actuation             --valve-open-ms / --valve-close-ms (datasheet or oscilloscope/LED test)

The deployed time debounce (500 ms) adds no delay to a change that is
accepted, but blocks any further change for 500 ms; its cost is evaluated in
e8_savings.py. It also reports the inter-camera capture skew within a batch (how "synchronised"
the five streams are) and, given the tractor speed, the distance travelled
during the total latency = the minimum lead of each camera ahead of its nozzles.

Example:
  python experiments/e7_e2e_latency.py --log logs/valves_*.csv --modbus-host 192.168.11.60 \
      --valve-open-ms 95 --valve-close-ms 60 --speed-kmh 6
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp.data import read_csv  # noqa: E402


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def modbus_rtt(host, port, coil, n, toggle=False):
    """Round-trip time of write_single_coil. By default each write repeats the value just read, so
    the valve never moves and the measurement can run next to the controller; toggle=True
    alternates the coil (bench only: it switches the real valves every 20 ms)."""
    from pyModbusTCP.client import ModbusClient
    c = ModbusClient(host=host, port=port, auto_open=True)
    read_write, write = [], []
    for i in range(n):
        t0 = time.perf_counter()
        cur = c.read_coils(coil, 1)
        t1 = time.perf_counter()
        c.write_single_coil(coil, bool(i % 2) if toggle else bool(cur and cur[0]))
        t2 = time.perf_counter()
        write.append((t2 - t1) * 1000)
        read_write.append((t2 - t0) * 1000)
        time.sleep(0.02)
    if toggle:
        c.write_single_coil(coil, True)  # leave the valve open (fail-safe = apply)
    return stats(write), stats(read_write)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", nargs="+", required=True)
    ap.add_argument("--modbus-host")
    ap.add_argument("--modbus-port", type=int, default=502)
    ap.add_argument("--modbus-coil", type=int, default=0)
    ap.add_argument("--modbus-n", type=int, default=500)
    ap.add_argument("--modbus-toggle", action="store_true",
                    help="alternate the coil instead of rewriting its value (bench only: moves the valve)")
    ap.add_argument("--valve-open-ms", type=float, default=float("nan"))
    ap.add_argument("--valve-close-ms", type=float, default=float("nan"))
    ap.add_argument("--speed-kmh", type=float, default=float("nan"))
    ap.add_argument("--out", default="experiments/results/e7_latency")
    args = ap.parse_args()

    rows = [r for f in args.log for g in glob.glob(f) for r in read_csv(g)]
    rows = [r for r in rows if r.get("method", "yolo") == "yolo" and r["t_capture_ns"] and r["t_received_ns"]]
    # int() keeps nanosecond precision (float64 loses it at epoch scale)
    ns = lambda k: np.array([int(r[k]) for r in rows], dtype=np.int64)  # noqa: E731
    tc, tp, tr, ti, tb = (ns(k) for k in ("t_capture_ns", "t_preprocessed_ns", "t_received_ns",
                                          "t_infer_done_ns", "t_logger_rx_ns"))
    stages = {
        "acquisition_buffer_preproc": stats((tp - tc) / 1e6),
        "transport_to_inference": stats((tr - tp) / 1e6),
        "inference": stats((ti - tr) / 1e6),
        "transport_to_controller": stats((tb - ti) / 1e6),
        "capture_to_controller_total": stats((tb - tc) / 1e6),
    }
    batches = {}
    for r in rows:
        batches.setdefault(r["t_preprocessed_ns"], []).append(int(r["t_capture_ns"]))
    stages["inter_camera_capture_skew"] = stats([(max(v) - min(v)) / 1e6 for v in batches.values() if len(v) > 1])

    modbus_mean = 0.0
    if args.modbus_host:
        w, rw = modbus_rtt(args.modbus_host, args.modbus_port, args.modbus_coil, args.modbus_n, args.modbus_toggle)
        stages["modbus_write"], stages["modbus_read_plus_write"] = w, rw
        modbus_mean = w["mean"]
    stages["valve_open"] = {"mean": args.valve_open_ms}
    stages["valve_close"] = {"mean": args.valve_close_ms}

    base = stages["capture_to_controller_total"].get("mean", float("nan")) + modbus_mean
    total = {"to_open_ms": base + args.valve_open_ms, "to_close_ms": base + args.valve_close_ms}
    if args.speed_kmh == args.speed_kmh:
        v = args.speed_kmh / 3.6 * 100  # cm/s
        total["lead_distance_cm_open"] = v * total["to_open_ms"] / 1000
        total["lead_distance_cm_close"] = v * total["to_close_ms"] / 1000
    span_s = (max(int(k) for k in batches) - min(int(k) for k in batches)) / 1e9 if len(batches) > 1 else 0
    report = {"n_decisions": len(rows), "n_batches": len(batches),
              "batch_rate_hz": (len(batches) - 1) / span_s if span_s else None,
              "stages_ms": stages, "total": total}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "latency.json").write_text(json.dumps(report, indent=2))
    for k, s in stages.items():
        if "p95" in s:
            print(f"  {k:<30} mean {s['mean']:8.2f} ms   p95 {s['p95']:8.2f}   n={s['n']}")
        else:
            print(f"  {k:<30} {s.get('mean', float('nan')):8.2f} ms")
    print("  TOTAL", json.dumps(total))


if __name__ == "__main__":
    main()
