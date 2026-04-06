from pathlib import Path

import labelbox as lb
import yaml

with open(Path(__file__).parent / "config.yml") as f:
    cfg = yaml.safe_load(f)["labelbox"]

client = lb.Client(api_key=cfg["api_key"])

for dataset in client.get_datasets():
    print("ID:", dataset.uid, "| Nombre:", dataset.name)
