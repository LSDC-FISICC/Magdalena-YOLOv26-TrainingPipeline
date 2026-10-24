#!/usr/bin/env python3
"""E4a - Download the Roboflow projects (one per pseudo-labelling variant) in folder format.

The API key is read from the ROBOFLOW_API_KEY environment variable. Each spec
is WORKSPACE/PROJECT[/VERSION] (version defaults to the latest one); slugs are
lower-case as in the Roboflow URL, e.g.
  laboratorios-workspace/sugarcanev1-wgpmk/1
  maximiliens-workspace/sugarcane_exg_hsv_lab_v3-2-100/1

Example:
  export ROBOFLOW_API_KEY=...
  python experiments/e4_roboflow_download.py --list maximiliens-workspace
  python experiments/e4_roboflow_download.py --out roboflow \
      maximiliens-workspace/sugarcane_exg_hsv_lab_v3 maximiliens-workspace/sugarcane_hsv_v4 ...
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="*")
    ap.add_argument("--list", metavar="WORKSPACE", help="print the projects of a workspace and exit")
    ap.add_argument("--out", default="roboflow")
    args = ap.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Set ROBOFLOW_API_KEY")
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)

    if args.list:
        for p in rf.workspace(args.list).project_list:
            print(f"{p['id']:<60} images={p.get('images', '?')}")
        return

    for spec in args.specs:
        parts = spec.strip("/").split("/")
        ws, proj = parts[0], parts[1]
        project = rf.workspace(ws).project(proj)
        version = int(parts[2]) if len(parts) > 2 else max(int(v.version) for v in project.versions())
        dst = Path(args.out) / proj
        print(f"Downloading {ws}/{proj} v{version} -> {dst}")
        project.version(version).download("folder", location=str(dst), overwrite=False)


if __name__ == "__main__":
    main()
