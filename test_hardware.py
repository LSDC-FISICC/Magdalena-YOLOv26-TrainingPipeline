#!/usr/bin/env python3
"""
Hardware diagnostic script for Jetson Orin Nano.
Tests TensorRT, CUDA, and model loading before running full inference.
"""

import os
import sys

print("\n" + "="*70)
print("JETSON ORIN NANO HARDWARE DIAGNOSTICS")
print("="*70 + "\n")

# Test 1: TensorRT
print("[1/6] Testing TensorRT...")
try:
    import tensorrt as trt
    print(f"  ✓ TensorRT {trt.__version__} available")
except ImportError as e:
    print(f"  ✗ TensorRT not available: {e}")
    sys.exit(1)

# Test 2: CUDA
print("\n[2/6] Testing CUDA Runtime...")
try:
    import pycuda.driver as cuda
    cuda.init()
    device_count = cuda.Device.count()
    print(f"  ✓ CUDA devices found: {device_count}")
    if device_count > 0:
        device = cuda.Device(0)
        print(f"    Device 0: {device.name()}")
        props = device.get_attributes()
        print(f"    Compute Capability: {props[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR]}.{props[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR   ]}")
except ImportError:
    print(f"  ⚠ pycuda not available (optional)")
except Exception as e:
    print(f"  ✗ CUDA error: {e}")

# Test 3: PyTorch
print("\n[3/6] Testing PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    print(f"    CUDA version: {torch.version.cuda if torch.version.cuda else 'None'}")
except ImportError as e:
    print(f"  ✗ PyTorch not available: {e}")

# Test 4: Ultralytics
print("\n[4/6] Testing Ultralytics...")
try:
    from ultralytics import YOLO
    print(f"  ✓ Ultralytics available")
except ImportError as e:
    print(f"  ✗ Ultralytics not available: {e}")
    sys.exit(1)

# Test 5: Find model
print("\n[5/6] Checking available models...")
weights_dir = "runs/classify/magdalena/plant_cls_v3/weights/"
if os.path.exists(weights_dir):
    models = os.listdir(weights_dir)
    print(f"  ✓ Models found in {weights_dir}:")
    for m in models:
        size = os.path.getsize(os.path.join(weights_dir, m)) / 1024**2
        print(f"    • {m} ({size:.1f} MB)")
else:
    print(f"  ✗ Models directory not found: {weights_dir}")
    sys.exit(1)

# Test 6: Try loading .engine
print("\n[6/6] Testing .engine loading...")
engine_path = "runs/classify/magdalena/plant_cls_v3/weights/best.engine"
if os.path.exists(engine_path):
    try:
        print(f"  Loading {engine_path}...")
        model = YOLO(engine_path, task='classify')
        print(f"  ✓ Engine loaded successfully!")
        
        # Try single inference
        print(f"\n  Testing single inference...")
        test_img = "dataset/test/plant/image_0.jpg"
        if not os.path.exists(test_img):
            # Find any image
            for root, dirs, files in os.walk("dataset/test"):
                for f in files:
                    if f.lower().endswith(('.jpg', '.png')):
                        test_img = os.path.join(root, f)
                        break
        
        if os.path.exists(test_img):
            print(f"  Using: {test_img}")
            results = model.predict(test_img, verbose=False, device='cpu', imgsz=320)
            result = results[0]
            pred_label = result.names[result.probs.top1]
            confidence = float(result.probs.top1conf)
            print(f"  ✓ Inference successful!")
            print(f"    Prediction: {pred_label} ({confidence:.2%})")
        else:
            print(f"  ⚠ No test image found")
            
    except Exception as e:
        print(f"  ✗ Error loading engine: {e}")
        print(f"  Falling back to .pt model...")
        
        pt_path = "runs/classify/magdalena/plant_cls_v3/weights/best.pt"
        if os.path.exists(pt_path):
            try:
                model = YOLO(pt_path, task='classify')
                print(f"  ✓ Loaded .pt instead")
            except Exception as e2:
                print(f"  ✗ Failed to load .pt: {e2}")
else:
    print(f"  ⚠ Engine not found at {engine_path}")
    pt_path = "runs/classify/magdalena/plant_cls_v3/weights/best.pt"
    if os.path.exists(pt_path):
        print(f"  Found .pt model instead, will use that")

print("\n" + "="*70)
print("DIAGNOSTICS COMPLETE")
print("="*70 + "\n")
