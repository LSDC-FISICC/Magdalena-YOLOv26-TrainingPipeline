import os
import labelbox as lb
import requests
import json
import yaml
from pathlib import Path

# config.yml (git-ignored) or, if absent, LABELBOX_API_KEY / LABELBOX_DATASET_ID
cfg_path = Path(__file__).parent / "config.yml"
if cfg_path.exists():
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["labelbox"]
else:
    cfg = {"api_key": os.environ["LABELBOX_API_KEY"], "dataset_id": os.environ.get("LABELBOX_DATASET_ID")}

client  = lb.Client(api_key=cfg["api_key"])
dataset = client.get_dataset(cfg["dataset_id"])

export_task = dataset.export(params={"data_row_details": True})
export_task.wait_till_done()

output_dir = Path("imagenes_labelbox")
output_dir.mkdir(exist_ok=True)

total  = 0
errors = 0

for item in export_task.get_buffered_stream():
    row      = item.json
    url      = row["data_row"]["row_data"]
    filename = row["data_row"]["external_id"]

    dest = output_dir / filename
    if dest.exists():
        continue

    # Sin header de autorización — las URLs de Labelbox ya llevan token en la URL
    response = requests.get(url)
    
    if response.status_code == 200:
        dest.write_bytes(response.content)
        total += 1
        print(f"[{total}] {filename}")
    else:
        print(f"Error {response.status_code}: {filename}")
        errors += 1

print(f"\nDescargadas: {total} | Errores: {errors}")