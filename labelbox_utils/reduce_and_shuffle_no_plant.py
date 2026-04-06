import random
from pathlib import Path

folder = Path("roboflow_upload/no_plant")
images = list(folder.glob("*.png"))

print(f"Imágenes actuales: {len(images)}")

# Seleccionar 2000 aleatoriamente para conservar
keep    = set(random.sample(images, 2000))
to_delete = [img for img in images if img not in keep]

for img in to_delete:
    img.unlink()

print(f"Eliminadas: {len(to_delete)}")
print(f"Quedan:     {len(list(folder.glob('*.png')))}")