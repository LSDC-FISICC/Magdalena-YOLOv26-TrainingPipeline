"""Shared helpers for the Magdalena publication experiments (see experiments/README.md)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("no_plant", "plant")   # Ultralytics sorts class folders alphabetically: 0=no_plant, 1=plant
IMG_H, IMG_W = 96, 320
