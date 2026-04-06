import labelbox as lb
import requests
import json
import yaml
from pathlib import Path

with open(Path(__file__).parent / "config.yml") as f:
    cfg = yaml.safe_load(f)["labelbox"]

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