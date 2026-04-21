#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    # Setup argparse for easy execution and optional overrides
    parser = argparse.ArgumentParser(description="Automated YOLO .pt to TensorRT .engine converter for Jetson Orin")
    parser.add_argument(
        "--model", 
        type=str, 
        default="runs/classify/magdalena/plant_cls_v3/weights/best.pt",
        help="Path to the YOLO .pt model (default: runs/classify/magdalena/plant_cls_v3/weights/best.pt)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=320,
        help="Static image size for the model (default: 320)"
    )
    args = parser.parse_args()

    pt_path = args.model
    if not os.path.exists(pt_path):
        print(f"Error: Model file not found at {pt_path}")
        sys.exit(1)

    # Determine paths for the intermediate ONNX and final Engine files
    pt_path_obj = Path(pt_path)
    onnx_path = pt_path_obj.with_suffix('.onnx')
    engine_path = pt_path_obj.with_suffix('.engine')

    print(f"\n{'='*70}")
    print(f"Jetson YOLO to TensorRT Converter")
    print(f"{'='*70}")
    print(f"Input model:   {pt_path}")
    print(f"Image size:    {args.imgsz}x{args.imgsz} (Static)")
    print(f"Target ONNX:   {onnx_path}")
    print(f"Target Engine: {engine_path}")
    print(f"{'='*70}\n")
    
    # ==========================================================
    # Step 1: Export to ONNX on CPU
    # Bypassing the GPU during export avoids the 
    # CUBLAS_STATUS_ALLOC_FAILED Out of Memory crash.
    # ==========================================================
    print(f"[Step 1/2] Exporting to ONNX via CPU...")
    yolo_cmd = [
        "yolo", "export", 
        f"model={pt_path}", 
        "format=onnx", 
        f"imgsz={args.imgsz}", 
        "device=cpu"
    ]
    
    print(f"Running: {' '.join(yolo_cmd)}")
    try:
        subprocess.run(yolo_cmd, check=True)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX file not generated at {onnx_path}")
        print("ONNX export successful!\n")
    except subprocess.CalledProcessError as e:
        print(f"\nError during ONNX export. See logs above.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

    # ==========================================================
    # Step 2: Compile to TensorRT Engine via trtexec
    # Using the native Jetson C++ toolchain for max optimization
    # ==========================================================
    print(f"[Step 2/2] Compiling TensorRT Engine via trtexec...")
    trtexec_path = "/usr/src/tensorrt/bin/trtexec"
    
    if not os.path.exists(trtexec_path):
        print(f"\nError: trtexec not found at {trtexec_path}.")
        print("Are you running this on a Jetson device with JetPack installed?")
        sys.exit(1)

    trt_cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16"
    ]
    
    print(f"Running: {' '.join(trt_cmd)}")
    try:
        subprocess.run(trt_cmd, check=True)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not generated at {engine_path}")
        print("\nTensorRT Engine compilation successful!")
    except subprocess.CalledProcessError as e:
        print(f"\nError during TensorRT compilation. See logs above.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

    # ==========================================================
    # Finish
    # ==========================================================
    print(f"\n{'='*70}")
    print(f"Conversion Complete!")
    print(f"Engine saved to: {engine_path}")
    print(f"\nFor inference, remember to use:")
    print(f"  model.predict(..., device=0, imgsz={args.imgsz})")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
