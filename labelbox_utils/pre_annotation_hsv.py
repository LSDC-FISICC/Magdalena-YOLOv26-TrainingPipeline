import cv2
import numpy as np
from pathlib import Path
import shutil

def get_label(img_path, threshold=0.001):
    img = cv2.imread(str(img_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)
    ratio = np.count_nonzero(mask) / mask.size
    return "plant" if ratio >= threshold else "no_plant"

source = Path("imagenes_labelbox")
images = sorted(source.glob("*.png"))

# Roboflow classification expects a folder structure:
# upload/plant/img1.png
# upload/no_plant/img2.png

output = Path("roboflow_upload")
(output / "plant").mkdir(parents=True, exist_ok=True)
(output / "no_plant").mkdir(parents=True, exist_ok=True)

plant_count    = 0
no_plant_count = 0

for img_path in images:
    label = get_label(img_path)
    shutil.copy(img_path, output / label / img_path.name)
    if label == "plant":
        plant_count += 1
    else:
        no_plant_count += 1

print(f"plant:    {plant_count}")
print(f"no_plant: {no_plant_count}")
print(f"Total:    {len(images)}")
print(f"\nFolder ready: roboflow_upload/")