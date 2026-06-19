import cv2
import numpy as np
from pathlib import Path
import shutil
import os



def calcul_exg(image) : 
    img = image.astype(np.float32) / 255.0
    R, G, B = img[:,:,2], img[:,:,1], img[:,:,0]
    total = R + G + B + 1e-6
    exg = 2*(G/total) - (R/total) - (B/total)
    return exg


def calcul_exgr(image):
    img = image.astype(np.float32) / 255.0
    R, G, B = img[:,:,2], img[:,:,1], img[:,:,0]
    total = R + G + B + 1e-6
    r, g, b = R/total, G/total, B/total
    exg  = 2*g - r - b
    exr  = 1.4*r - g        
    return exg - exr

def get_label(img_path, exg_thresh=0.02, exgr_thresh=0.0, ratio_min=0.0005, hsv_thresh=0.001, v_min = 60):
    img = cv2.imread(str(img_path))
    if img is None:
        return "no_plant"
    h, w = img.shape[:2]

    # ── Masque ExG / not used here (generate some fake positive) ────────────────────────────────────────────────
    exg      = calcul_exg(img)
    mask_exg = (exg > exg_thresh).astype(np.uint8) * 255

    # ── Masque ExGR ────────────────────────────────────────────────

    exgr = calcul_exgr(img)
    mask_exgr = (exgr > exgr_thresh).astype(np.uint8) * 255

    # ── Masque HSV ────────────────────────────────────────────────
    hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_hsv= cv2.inRange(hsv, np.array([35,40,40]), np.array([85,255,255]))

    # ── Masque ExG without darker pixels ────────────────────────────────────────────────

    v_mask = (hsv[:,:,2] > v_min).astype(np.uint8) * 255
    mask_exg = cv2.bitwise_and(mask_exg, v_mask)
    mask_exgr = cv2.bitwise_and(mask_exgr, v_mask)

    # ── Fusion ────────────────────────────────────────────────────
    #mask = cv2.bitwise_or(mask_exg, mask_hsv)
    mask = mask_hsv

    # ── Morphologie CLOSE  ─────────────────────────────
    k5   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

    # ── Morphologie OPEN  ─────────────────────────────
    """
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    mask_no_hlines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_h)
    mask_no_vlines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_v)
    mask = cv2.bitwise_and(mask_no_hlines, mask_no_vlines)
    """
    # ── Technique 1 : HSV ratio─────────
    ratio_global = np.count_nonzero(mask) / mask.size
    if ratio_global >= hsv_thresh:
        return "plant"

    # ── Technique 2 : contours with fusioned mask ────────────────
    mask_bright_plant = cv2.inRange(hsv, np.array([25, 15, 40]), np.array([90, 255, 255]))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channel_a = lab[:, :, 1]
    channel_b = lab[:, :, 2]
    mask_lab_green = cv2.inRange(channel_a, 0, 125)
    mask_lab_yellow = cv2.inRange(channel_b, 126, 150) # 0, 145 

    mask_lab = cv2.bitwise_and(mask_lab_green, mask_lab_yellow)
    mask_bright_plant = cv2.bitwise_or(mask_bright_plant, mask_lab)
    mask_for_contours = cv2.bitwise_or(mask_bright_plant, mask_exgr) #exgr only to generate less fake positive
    mask_for_contours = cv2.morphologyEx(mask_for_contours, cv2.MORPH_CLOSE, k5)

    contours, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "no_plant"

    contours = [c for c in contours if cv2.contourArea(c) > 10]
    if not contours:
        return "no_plant"

    max_area_ratio = max((cv2.contourArea(c) for c in contours), default=0)  / (h * w)
    return "plant" if max_area_ratio >= ratio_min else "no_plant"


source = Path("imagenes_labelbox")
print(source.exists())
images = sorted(source.glob("*.png"))
print(len(images))

output = Path("roboflow_upload_exg_hsv_lab_v3-3")
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
print(f"\nFolder ready: roboflow_upload_exg_hsv_lab/")