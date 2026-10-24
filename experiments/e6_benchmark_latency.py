#!/usr/bin/env python3
"""E6 - Inference latency, throughput and energy on the Jetson Orin Nano.

Fills "Throughput reached [XX] images per second" and the inference term of the
latency decomposition. The engine is timed through the TensorRT Python API with
the same dynamic-batch bindings the C++ node uses ("images" -> "output0"), so
the numbers exclude Python/Ultralytics overhead. For each batch size 1..5:
  * --iters timed executions after --warmup, reported as mean/std/p50/p95/p99 (ms per batch),
  * throughput = batch / mean latency (images/s), and frames-per-second per camera
    when the five cameras are batched together,
  * average board power from tegrastats (VDD_IN) and energy per image (mJ),
  * host->device copy time, and the OpenCV crop/resize/normalise cost of 5 frames.
The nvpmodel power mode and jetson_clocks state are stored with the results:
report them in the paper (e.g. 15 W / MAXN SUPER, clocks locked).

Examples (on the Jetson):
  sudo jetson_clocks
  python experiments/e6_benchmark_latency.py --engine best_fp16.engine --tag yolo26n_fp16
  python experiments/e6_benchmark_latency.py --engine best_fp32.engine --tag yolo26n_fp32
  # any ONNX model (e.g. MobileNetV3 from e4 --export-onnx) can be compiled first:
  /usr/src/tensorrt/bin/trtexec --onnx=m.onnx --saveEngine=m.engine --fp16 \
      --minShapes=images:1x3x96x320 --optShapes=images:5x3x96x320 --maxShapes=images:5x3x96x320
"""

import argparse
import json
import platform
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


class Tegrastats:
    """Samples VDD_IN (total board power, mW) from tegrastats in a background thread."""

    def __init__(self, interval_ms=100):
        self.interval = interval_ms
        self.samples, self.proc, self.thread = [], None, None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval)],
                                         stdout=subprocess.PIPE, text=True)
        except FileNotFoundError:
            return self
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            m = re.search(r"VDD_IN (\d+)mW", line)
            if m:
                self.samples.append(int(m.group(1)))

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()

    def mean_w(self):
        return float(np.mean(self.samples)) / 1000 if self.samples else float("nan")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def preprocess_cost(n_cams=5, iters=200, src_hw=(360, 640)):
    """Centre crop 320x96 + float conversion of 5 frames, as cameraNode does."""
    import cv2
    frames = [np.random.randint(0, 255, (*src_hw, 3), np.uint8) for _ in range(n_cams)]
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        batch = []
        for f in frames:
            h, w = f.shape[:2]
            crop = f[h // 2 - 48:h // 2 + 48, w // 2 - 160:w // 2 + 160]
            batch.append(cv2.resize(crop, (320, 96)).astype(np.float32) / 255.0)
        np.ascontiguousarray(np.stack(batch).transpose(0, 3, 1, 2)[:, ::-1])
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(t)), float(np.percentile(t, 95))


def bench_trt(engine_path, batches, iters, warmup):
    import tensorrt as trt
    import torch
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as fh:
        engine = trt.Runtime(logger).deserialize_cuda_engine(fh.read())
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()
    in_name, out_name = "images", "output0"
    results = []
    for b in batches:
        x_host = torch.rand(b, 3, 96, 320).pin_memory()
        x = torch.empty((b, 3, 96, 320), device="cuda")
        ctx.set_input_shape(in_name, (b, 3, 96, 320))
        y = torch.empty(tuple(ctx.get_tensor_shape(out_name)), device="cuda")
        ctx.set_tensor_address(in_name, x.data_ptr())
        ctx.set_tensor_address(out_name, y.data_ptr())
        lat, h2d = [], []
        tg = Tegrastats()
        for i in range(warmup + iters):
            if i == warmup:
                tg.__enter__()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                x.copy_(x_host, non_blocking=True)
            stream.synchronize()
            t1 = time.perf_counter()
            ctx.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            t2 = time.perf_counter()
            if i >= warmup:
                h2d.append((t1 - t0) * 1000)
                lat.append((t2 - t1) * 1000)
        tg.__exit__()
        lat = np.array(lat)
        power = tg.mean_w()
        results.append({
            "batch": b, "mean_ms": lat.mean(), "std_ms": lat.std(), "p50_ms": np.percentile(lat, 50),
            "p95_ms": np.percentile(lat, 95), "p99_ms": np.percentile(lat, 99),
            "h2d_ms": float(np.mean(h2d)), "throughput_img_s": b * 1000 / lat.mean(),
            "max_fps_per_camera": 1000 / lat.mean(), "power_w": power,
            "energy_mj_per_img": power * lat.mean() / b if power == power else float("nan"),
        })
        r = results[-1]
        print(f"  batch {b}: {r['mean_ms']:.3f} ms (p95 {r['p95_ms']:.3f})  "
              f"{r['throughput_img_s']:.0f} img/s  {r['power_w']:.2f} W  {r['energy_mj_per_img']:.2f} mJ/img")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--tag", required=True, help="name of this configuration in the results")
    ap.add_argument("--batches", default="1,2,3,4,5")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", default="experiments/results/e6_latency")
    args = ap.parse_args()

    env = {"host": platform.node(), "nvpmodel": sh("nvpmodel -q"), "jetson_clocks": sh("jetson_clocks --show"),
           "l4t": sh("cat /etc/nv_tegra_release"), "engine": args.engine}
    print("Power mode:", env["nvpmodel"].splitlines()[:1])
    res = bench_trt(args.engine, [int(b) for b in args.batches.split(",")], args.iters, args.warmup)
    pre_mean, pre_p95 = preprocess_cost()
    print(f"  preprocessing (5 frames, CPU): {pre_mean:.3f} ms (p95 {pre_p95:.3f})")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(
        {"env": env, "results": res, "preprocess_5cams_ms": {"mean": pre_mean, "p95": pre_p95}},
        indent=2, default=float))
    print("Saved", out / f"{args.tag}.json")


if __name__ == "__main__":
    main()
