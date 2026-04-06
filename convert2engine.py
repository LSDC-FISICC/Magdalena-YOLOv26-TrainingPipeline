import argparse
from ultralytics import YOLO

DEFAULT_WEIGHTS = "/home/julioefajardo/Magdalena-YOLOv26-TrainingPipeline-main/runs/classify/magdalena/plant_cls_v3/weights/best.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to .pt weights file")
args = parser.parse_args()

model = YOLO(args.weights)
model.export(
    format = "engine",
    imgsz  = 320,
    device = 0,
    half   = True,
)