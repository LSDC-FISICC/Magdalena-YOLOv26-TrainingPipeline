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
DATASET_ROOT = "dataset2/test"

# Model priority: ONNX (CPU, no CUDA issues) > PT (CPU) > Engine (GPU, CUDA required)
# On Jetson with CUDA driver issues, ONNX is most reliable
MODEL_PRIORITY = [
    #"runs/classify/magdalena/plant_cls_v4/weights/best.onnx",  # CPU inference - most reliable
    #"runs/classify/magdalena/plant_cls_v4/weights/best.pt",    # CPU inference - fallback
    "runs/classify/magdalena/plant_cls_v4/weights/best.engine", # GPU only - if CUDA works
]

BATCH_SIZE = 5 

WARMUP_RUNS = 5

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


# Flatten all images into a list of (true_label, img_name, img_path)
all_images = []
for true_label, images in sorted(test_images.items()):
    for img_name in images:
        img_path = os.path.join(DATASET_ROOT, true_label, img_name)
        all_images.append((true_label, img_name, img_path))

# ============================================================================
# Run Inference
# ============================================================================
print(f"\n{'='*70}")
print(f"Running Inference (batch_size={BATCH_SIZE})...")
print(f"{'='*70}\n")

correct = 0
incorrect = 0
predictions = []
batch_times = []
errors_list = []

start_total = time.perf_counter()

dummy = [all_images[0][2]] * BATCH_SIZE

for _ in range(WARMUP_RUNS):
        model.predict(dummy, verbose=False, device=0, imgsz=[96, 320])
print("✓ Warm-up done")

for batch_idx, batch_start in enumerate(range(0, len(all_images), BATCH_SIZE)):
    batch = all_images[batch_start:batch_start + BATCH_SIZE]
    batch_paths = [item[2] for item in batch]

    try:
        start = time.perf_counter()
        results = model.predict(batch_paths, verbose=False, device=0, imgsz=[96, 320])
        elapsed = time.perf_counter() - start
        batch_times.append(elapsed)

        for (true_label, img_name, img_path), result in zip(batch, results):
            if hasattr(result, 'probs') and result.probs is not None:
                pred_idx = result.probs.top1
                pred_label = result.names[pred_idx]
                confidence = float(result.probs.top1conf)
            else:
                print(f"⚠ Unexpected result format for {img_name}")
                continue

            if pred_label == 'class0':
                pred_label = 'no_plant'
            elif pred_label == 'class1':
                pred_label = 'plant'

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
            })

            processed = correct + incorrect
            if processed % 10 == 0:
                current_acc = 100 * correct / processed
                print(f"  [{processed:3d}/{total_images}] {current_acc:5.1f}% | "
                      f"{symbol} {img_name[:30]:30s} → {pred_label:10s} "
                      f"({confidence:5.1%})")

        print(f"  Batch {batch_idx+1:3d} [{len(batch)} images]: {elapsed*1000:.2f} ms "
              f"({elapsed*1000/len(batch):.2f} ms/img)")

    except Exception as e:
        for (true_label, img_name, img_path) in batch:
            incorrect += 1
            errors_list.append((img_name, str(e)))
        print(f"  ✗ Batch {batch_idx+1} error: {str(e)[:100]}")

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
mean_batch_time = np.mean(batch_times) * 1000 if batch_times else 0
std_batch_time = np.std(batch_times) * 1000 if batch_times else 0
throughput = total_processed / total_time

print(f"  Accuracy:              {accuracy:6.2f}% ({correct}/{total_processed})")
print(f"  Mean Batch Latency:    {mean_batch_time:6.2f} ms ± {std_batch_time:5.2f} ms")
print(f"  Mean Latency/image:    {mean_batch_time/BATCH_SIZE:6.2f} ms")
print(f"  Throughput:            {throughput:6.1f} images/sec")
print(f"  Total Time:            {total_time:6.2f} sec")
print(f"  Errors:                {len(errors_list):6d}")

if incorrect > 0 and len(predictions) > 0:
    misclassified = [p for p in predictions if not p['correct']]
    print(f"\n  Misclassified ({len(misclassified)}):")
    print("  showing first 5 misclassifications:")
    for pred in misclassified[:5]:
        print(f"    • {pred['image'][:16]:16s}")
        print(f"      True: {pred['true']:10s} | Pred: {pred['pred']:10s} "
              f"({pred['confidence']:5.1%})")

if errors_list:
    print(f"\n  Processing Errors ({len(errors_list)}):")
    for img_name, error in errors_list[:5]:
        print(f"    • {img_name}: {error[:50]}")

print(f"\n{'='*70}\n")

sys.exit(0 if len(errors_list) == 0 else 1)
