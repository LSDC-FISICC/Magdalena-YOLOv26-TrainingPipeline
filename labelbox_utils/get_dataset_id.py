import os
from pathlib import Path

import labelbox as lb
import yaml

# config.yml (git-ignored) or, if absent, LABELBOX_API_KEY / LABELBOX_DATASET_ID
cfg_path = Path(__file__).parent / "config.yml"
if cfg_path.exists():
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["labelbox"]
else:
    cfg = {"api_key": os.environ["LABELBOX_API_KEY"], "dataset_id": os.environ.get("LABELBOX_DATASET_ID")}

client = lb.Client(api_key=cfg["api_key"])

for dataset in client.get_datasets():
    print("ID:", dataset.uid, "| Nombre:", dataset.name)
