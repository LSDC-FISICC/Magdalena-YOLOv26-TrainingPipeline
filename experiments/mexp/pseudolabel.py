"""Classical colour-index labellers used as pseudo-label sources and as baselines.

The HSV and cascade (HSV + LAB + ExGR) labellers are imported from
``labelbox_utils`` so the baseline is exactly the function that produced the
training labels. ``cvpipeline_exg`` reproduces the RGB index path of
Magdalena-CV-Pipeline (HSV green mask -> ExG binarisation -> mean > threshold).
NDVI is not reproducible offline: the dataset holds RGB frames only.
"""

import importlib.util

import cv2
import numpy as np

from . import REPO_ROOT


def _load(name):
    path = REPO_ROOT / "labelbox_utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hsv = _load("pre_annotation_hsv")
_cascade = _load("pre_annotation_exg_hsv_lab")


def hsv(path):
    return _hsv.get_label(path)


def cascade(path):
    return _cascade.get_label(path)


def cvpipeline_exg(path, egx_threshold=0.501, mean_threshold=0.02):
    img = cv2.imread(str(path))
    if img is None:
        return "no_plant"
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, np.array([40, 75, 75]), np.array([80, 255, 255]))
    img = cv2.bitwise_and(img, img, mask=mask)
    b, g, r = [img[:, :, i].astype(np.float32) for i in range(3)]
    exg = ((2 * g - r - b) + 510) / 1020
    ratio = float((exg > egx_threshold).mean())
    return "plant" if ratio > mean_threshold else "no_plant"


LABELERS = {"hsv": hsv, "cascade_hsv_lab_exgr": cascade, "cvpipeline_exg": cvpipeline_exg}
