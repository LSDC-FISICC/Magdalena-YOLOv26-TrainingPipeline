#!/usr/bin/env python3
"""
Simple inference script for Jetson Orin Nano.
Runs full inference on test dataset with robust error handling.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ============================================================================
# Configuration
# ============================================================================
DATASET_ROOT = "dataset/test"

# Model priority: ONNX (CPU, no CUDA issues) > PT (CPU) > Engine (GPU, CUDA required)
# On Jetson with CUDA driver issues, ONNX is most reliable
MODEL_PRIORITY = [
    #"runs/classify/magdalena/plant_cls_v3/weights/best.onnx",  # CPU inference - most reliable
    #"runs/classify/magdalena/plant_cls_v3/weights/best.pt",    # CPU inference - fallback
    "runs/classify/magdalena/plant_cls_v3/weights/best.engine", # GPU only - if CUDA works
]

# ============================================================================
# Find Model
# ============================================================================
print(f"\n{'='*70}")
print("Finding Model...")
print(f"{'='*70}")

MODEL_PATH = None
for candidate in MODEL_PRIORITY:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

if not MODEL_PATH:
    print("❌ No model found at:")
    for candidate in MODEL_PRIORITY:
        print(f"   {candidate}")
    sys.exit(1)

model_format = MODEL_PATH.split('.')[-1].upper()
print(f"✓ Using model: {MODEL_PATH}")
print(f"  Format: {model_format}")

# ============================================================================
# Load Model
# ============================================================================
print(f"\n{'='*70}")
print("Loading Model...")
print(f"{'='*70}")

try:
    # Force CPU device to avoid CUDA issues
    model = YOLO(MODEL_PATH, task='classify')
    print(f"✓ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# ============================================================================
# Prepare Test Data
# ============================================================================
print(f"\n{'='*70}")
print("Scanning Test Dataset...")
print(f"{'='*70}")

test_images = defaultdict(list)
for class_name in ["plant", "no_plant"]:
    class_dir = os.path.join(DATASET_ROOT, class_name)
    if os.path.exists(class_dir):
        images = sorted([
            f for f in os.listdir(class_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        test_images[class_name] = images
        print(f"  {class_name:12} {len(images):3d} images")

total_images = sum(len(imgs) for imgs in test_images.values())
print(f"  {'─'*30}")
print(f"  {'Total':12} {total_images:3d} images")

if total_images == 0:
    print("❌ No test images found!")
    sys.exit(1)

# ============================================================================
# Run Inference
# ============================================================================
print(f"\n{'='*70}")
print("Running Inference...")
print(f"{'='*70}\n")

correct = 0
incorrect = 0
predictions = []
inference_times = []
errors_list = []

start_total = time.perf_counter()

for true_label, images in sorted(test_images.items()):
    for idx, img_name in enumerate(images):
        img_path = os.path.join(DATASET_ROOT, true_label, img_name)
        
        try:
            # Inference with CPU device
            start = time.perf_counter()
            results = model.predict(img_path, verbose=False, device=0, imgsz=320)
            elapsed = time.perf_counter() - start
            inference_times.append(elapsed)
            
            # Extract prediction
            result = results[0]
            
            # Handle different output formats
            if hasattr(result, 'probs') and result.probs is not None:
                pred_idx = result.probs.top1
                pred_label = result.names[pred_idx]
                confidence = float(result.probs.top1conf)
            else:
                # Fallback
                print(f"⚠ Unexpected result format for {img_name}")
                continue
            
            #change class0 to no_plant and class1 to plant
            if pred_label == 'class0':
                pred_label = 'no_plant'
            elif pred_label == 'class1':
                pred_label = 'plant'    
            # Check correctness
            is_correct = pred_label == true_label
            if is_correct:
                correct += 1
                symbol = "✓"
            else:
                incorrect += 1
                symbol = "✗"
            
            predictions.append({
                'image': img_name,
                'true': true_label,
                'pred': pred_label,
                'confidence': confidence,
                'correct': is_correct,
                'time': elapsed
            })
            
            # Print progress every 50 images
            processed = correct + incorrect
            if processed % 10 == 0:
                current_acc = 100 * correct / processed
                print(f"  [{processed:3d}/{total_images}] {current_acc:5.1f}% | "
                      f"{symbol} {img_name[:30]:30s} → {pred_label:10s} "
                      f"({confidence:5.1%}) {elapsed*1000:6.2f}ms")
        
        except Exception as e:
            incorrect += 1
            errors_list.append((img_name, str(e)))
            print(f"  ✗ Error on {img_name[:15]}: {str(e)[:100]}")

total_time = time.perf_counter() - start_total

# ============================================================================
# Results Summary
# ============================================================================
print(f"\n{'='*70}")
print("Inference Results")
print(f"{'='*70}\n")

total_processed = correct + incorrect
if total_processed == 0:
    print("❌ No images processed!")
    sys.exit(1)

accuracy = 100 * correct / total_processed
mean_latency = np.mean(inference_times) * 1000 if inference_times else 0
std_latency = np.std(inference_times) * 1000 if inference_times else 0
throughput = total_processed / total_time

print(f"  Accuracy:              {accuracy:6.2f}% ({correct}/{total_processed})")
print(f"  Mean Latency:          {mean_latency:6.2f} ms ± {std_latency:5.2f} ms")
print(f"  Throughput:            {throughput:6.1f} images/sec")
print(f"  Total Time:            {total_time:6.2f} sec")
print(f"  Errors:                {len(errors_list):6d}")

if incorrect > 0 and len(predictions) > 0:
    print(f"\n  Misclassified ({len([p for p in predictions if not p['correct']])}):")
    print("showing first 5 misclassifications:")
    for pred in predictions[:5]:
        if not pred['correct']:
            print(f"    • {pred['image'][:40]:40s}")
            print(f"      True: {pred['true']:10s} | Pred: {pred['pred']:10s} "
                  f"({pred['confidence']:5.1%})")

if errors_list:
    print(f"\n  Processing Errors ({len(errors_list)}):")
    for img_name, error in errors_list[:5]:  # Show first 5 errors
        print(f"    • {img_name}: {error[:50]}")

print(f"\n{'='*70}\n")

# Exit with status
sys.exit(0 if len(errors_list) == 0 else 1)
