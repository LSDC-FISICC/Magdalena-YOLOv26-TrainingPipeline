#!/usr/bin/env python3
"""E3/E4/E5/E11 with three seeds and the full protocol on a GPU workstation (DGX Spark or an RTX laptop).

Five steps, each resumable (a finished training writes <run>.done.json and is not repeated):

  prep     download the Roboflow variants that are not in git (needs ROBOFLOW_API_KEY),
           build the E3 pseudo-labelled set from a frame,label CSV, re-split every dataset
           with the temporal block split, and write the provisional GT (block-test frames
           with the curated labels) used when no human GT is given;
  smoke    one-epoch YOLO26n-cls and MobileNetV3 runs: checks that the GPU is used;
  run      the trainings, --jobs at a time (E3: one job per seed; E4/E5/E11: one per model and seed;
           E11 = YOLO26n-cls with online motion-blur and colour augmentation, run with --only e11);
  summary  mean +- std over seeds for every configuration -> experiments/results/gpu/<tag>/summary.csv;
  evaluate re-scores every trained model on another GT (the human one) without retraining.

All comparisons of the paper must come from the same machine: do not mix GPUs.
Hyper-parameters come from configs/train.yaml (30 epochs, patience 5, batch 64).

Example:
  export ROBOFLOW_API_KEY=...
  python experiments/run_gpu_suite.py prep --pseudo-csv e3_pseudo_labels.csv
  python experiments/run_gpu_suite.py smoke
  python experiments/run_gpu_suite.py run --jobs 3            # DGX Spark; use --jobs 1 on a laptop
  python experiments/run_gpu_suite.py summary
  # later, with the human GT (gt.csv from e1_agreement.py, paths valid on this machine):
  python experiments/run_gpu_suite.py evaluate --gt gt_humano.csv --tag humangt
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mexp import REPO_ROOT  # noqa: E402
from mexp.data import link_or_copy, list_folder_dataset, resplit, write_csv  # noqa: E402

PY = sys.executable
EXP = REPO_ROOT / "experiments"
WORK = EXP / "work" / "gpu"
RES = EXP / "results" / "gpu"
SEEDS = (0, 1, 2)

# Labelling variants (E4). The curated cascade (dataset2) and SugarCaneV1 (dataset) are in git;
# the others are downloaded from Roboflow.
ROBOFLOW = {
    "hsv_v1": "laboratorios-workspace/sugarcanehsv_v1/4",
    "exg_v1": "laboratorios-workspace/sugarcaneexg_v1/1",
    "exg_hsv_v2": "laboratorios-workspace/sugarcane_exg_hsv_v2/1",
    "cascade_v3": "laboratorios-workspace/sugarcane_exg_hsv_lab_v3/1",
}
LOCAL = {"curated": REPO_ROOT / "dataset2", "manual_v1": REPO_ROOT / "dataset"}
E4_DATASETS = ["hsv_v1", "exg_v1", "manual_v1", "exg_hsv_v2", "cascade_v3"]  # + curated, shared with E5
E5_ARCHS = ["yolo26n-cls.pt", "yolo26s-cls.pt", "mobilenet_v3_small", "efficientnet_b0", "resnet18"]
E3_FRACTIONS = "0,10,25,50,100"
E11_DATASETS = ["curated", "cascade_v3"]


def source_dir(name):
    if name in LOCAL:
        return LOCAL[name]
    return REPO_ROOT / "roboflow" / ROBOFLOW[name].split("/")[1]


def split_dir(name):
    return WORK / "data" / name


def provisional_gt():
    return WORK / "provisional_gt_blocktest_curated.csv"


def prep(args):
    missing = [n for n in ROBOFLOW if not source_dir(n).exists()]
    if missing:
        if not os.environ.get("ROBOFLOW_API_KEY"):
            sys.exit(f"Missing {missing}: set ROBOFLOW_API_KEY or copy them into roboflow/")
        subprocess.run([PY, str(EXP / "e4_roboflow_download.py"), "--out", str(REPO_ROOT / "roboflow"),
                        *[ROBOFLOW[n] for n in missing]], check=True)
    for name in ["curated", *E4_DATASETS]:
        if not split_dir(name).exists():
            print(f"block split: {name}")
            resplit(source_dir(name), split_dir(name))
    # Provisional GT: the held-out test blocks of the curated dataset (one image per frame)
    rows = [{"path": str(Path(r["path"]).resolve()), "label": r["label"]}
            for r in list_folder_dataset(split_dir("curated"), "test")]
    write_csv(provisional_gt(), rows, ["path", "label"])
    print(f"provisional GT: {len(rows)} images -> {provisional_gt()}")
    # E3 pseudo-labelled set: the curated images relabelled with the uncurated cascade label
    if args.pseudo_csv:
        lab = {int(r["frame"]): r["label"] for r in csv.DictReader(open(args.pseudo_csv))}
        n = 0
        for r in list_folder_dataset(LOCAL["curated"]):
            if r["frame"] in lab:
                link_or_copy(r["path"], WORK / "e3_pseudo" / r["split"] / lab[r["frame"]] / Path(r["path"]).name)
                n += 1
        print(f"E3 pseudo set: {n} images, {len(lab)} labelled frames -> {WORK / 'e3_pseudo'}")
    elif not (WORK / "e3_pseudo").exists():
        print("WARNING: no --pseudo-csv; E3 will be skipped")


def e45_cmd(exp, dataset, arch, seed, gt, tag, device, extra=()):
    aname = Path(arch).stem + ("_aug" if "--aug" in extra else "")
    return [PY, str(EXP / "e4_train_compare.py"), "--dataset", f"{dataset}={split_dir(dataset)}",
            "--arch", arch, "--seeds", str(seed), "--gt", str(gt), "--device", device,
            "--work", str(WORK / exp), "--out", str(RES / tag / exp / f"{dataset}__{aname}__s{seed}"), *extra]


def jobs(args, gt):
    out = []
    if (WORK / "e3_pseudo").exists() and "e3" in args.only:
        for s in SEEDS:  # one job per seed; its five fractions run in sequence
            out.append(("e3", f"seed{s}", [
                PY, str(EXP / "e3_ablation_curation.py"), "--pseudo", str(WORK / "e3_pseudo"),
                "--curated", str(LOCAL["curated"]), "--gt", str(gt), "--fractions", E3_FRACTIONS,
                "--seeds", str(s), "--device", args.device, "--work", str(WORK / "e3"),
                "--out", str(RES / args.tag / "e3" / f"s{s}")]))
    if "e5" in args.only:
        for a in E5_ARCHS:
            for s in SEEDS:
                out.append(("e5", f"{Path(a).stem}_s{s}",
                            e45_cmd("e5", "curated", a, s, gt, args.tag, args.device, ["--export-onnx"])))
    if "e4" in args.only:
        for d in E4_DATASETS:
            for s in SEEDS:
                out.append(("e4", f"{d}_s{s}", e45_cmd("e4", d, "yolo26n-cls.pt", s, gt, args.tag, args.device)))
    if "e11" in args.only:  # online blur/colour augmentation, curated and uncurated cascade labels
        for d in E11_DATASETS:
            for s in SEEDS:
                out.append(("e11", f"{d}_s{s}",
                            e45_cmd("e11", d, "yolo26n-cls.pt", s, gt, args.tag, args.device, ["--aug", "--export-onnx"])))
    return out


def run_one(job, log_dir):
    exp, name, cmd = job
    log = log_dir / f"{exp}_{name}.log"
    t0 = time.time()
    with open(log, "a") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO_ROOT).returncode
    print(f"[{time.strftime('%H:%M:%S')}] {exp} {name}: {'ok' if rc == 0 else f'FAILED ({rc}), see {log}'}"
          f" ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return rc


def run(args):
    gt = Path(args.gt) if args.gt else provisional_gt()
    if not gt.exists():
        sys.exit(f"{gt} not found: run prep first")
    js = jobs(args, gt)
    log_dir = WORK / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(js)} jobs, {args.jobs} at a time, GT = {gt}")
    with ThreadPoolExecutor(args.jobs) as ex:
        rcs = list(ex.map(lambda j: run_one(j, log_dir), js))
    print(f"done: {rcs.count(0)}/{len(rcs)} ok")


def smoke(args):
    gt = provisional_gt()
    if not gt.exists():
        sys.exit("run prep first")
    for f in (WORK / "smoke").glob("*.done.json"):  # always train again
        f.unlink()
    import torch
    print("torch", torch.__version__, "CUDA", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    for arch in ("yolo26n-cls.pt", "mobilenet_v3_small"):
        cmd = [PY, str(EXP / "e4_train_compare.py"), "--dataset", f"curated={split_dir('curated')}",
               "--arch", arch, "--seeds", "0", "--gt", str(gt), "--device", args.device, "--epochs", "1",
               "--work", str(WORK / "smoke"), "--out", str(RES / "smoke" / Path(arch).stem)]
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        print(f"SMOKE {arch}: {'ok' if rc == 0 else 'FAILED'} in {time.time() - t0:.0f} s")
        if rc:
            sys.exit(rc)


def done_records():
    """[(experiment, group_a, group_b, record)] for every finished E3/E4/E5/E11 training."""
    out = []
    for exp in ("e3", "e4", "e5", "e11"):
        for f in sorted((WORK / exp).glob("*.done.json")):
            r = json.loads(f.read_text())
            if exp == "e3":
                out.append((exp, "curated_fraction", f"{float(r['fraction']):g}%", r))
            else:
                out.append((exp, r["dataset"], r["arch"], r))
    return out


def summarise(records, path):
    metrics = ("accuracy", "recall", "specificity", "mcc")
    groups = defaultdict(list)
    for exp, a, b, r in records:
        groups[(exp, a, b)].append(r)
    rows = []
    for (exp, a, b), rs in sorted(groups.items()):
        row = {"experiment": exp, "dataset_or_factor": a, "arch_or_level": b, "seeds": len(rs)}
        for m in metrics:
            v = np.array([float(r[m]) for r in rs])
            row[f"{m}_mean"] = round(float(v.mean()), 4)
            row[f"{m}_std"] = round(float(v.std(ddof=1)), 4) if len(v) > 1 else ""
        rows.append(row)
        print(f"{exp}  {a:<18} {b:<20} n={len(rs)}  acc {100 * row['accuracy_mean']:.1f}"
              + (f" +- {100 * row['accuracy_std']:.1f}" if row["accuracy_std"] != "" else ""))
    if rows:
        write_csv(path, rows)
        print(f"-> {path}")


def summary(args):
    summarise(done_records(), RES / args.tag / "summary.csv")


def evaluate(args):
    if not args.gt:
        sys.exit("evaluate needs --gt")
    from mexp.data import read_csv
    from mexp.metrics import summary as metrics
    from mexp.models import Predictor
    gt = read_csv(args.gt)
    paths, y = [r["path"] for r in gt], [r["label"] for r in gt]
    records, per_run = [], []
    for exp, a, b, r in done_records():
        weights = Path(r["weights"])
        weights = weights if weights.is_absolute() else REPO_ROOT / weights
        pred, _ = Predictor(weights).predict(paths)
        m = dict(metrics(y, pred), seed=r.get("seed"), weights=str(weights))
        records.append((exp, a, b, m))
        per_run.append({"experiment": exp, "dataset_or_factor": a, "arch_or_level": b, **m})
    write_csv(RES / args.tag / "runs.csv", per_run, list(dict.fromkeys(k for r in per_run for k in r)))
    summarise(records, RES / args.tag / "summary.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prep", "smoke", "run", "summary", "evaluate"])
    ap.add_argument("--pseudo-csv", help="prep: frame,label CSV of the uncurated cascade (E3)")
    ap.add_argument("--gt", help="run/evaluate: GT csv (run defaults to the provisional block-test GT)")
    ap.add_argument("--tag", default="provisional", help="results sub-folder, e.g. humangt")
    ap.add_argument("--jobs", type=int, default=1, help="trainings in parallel (3 on a DGX Spark)")
    ap.add_argument("--only", default="e3,e4,e5")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    {"prep": prep, "smoke": smoke, "run": run, "summary": summary, "evaluate": evaluate}[args.step](args)


if __name__ == "__main__":
    main()
