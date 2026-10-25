#!/bin/bash
# E6 on the Jetson Orin Nano (reComputer J3011): build the YOLO26n-cls TensorRT engines on the
# device and time them with batch 1..5. Engines are device-specific, so build them here.
#
#   bash experiments/jetson_e6.sh [model.onnx] [tag]
#
# Defaults to the YOLO26n-cls model trained on the leak-free split. Any other ONNX with the
# same "images" -> "output0" bindings (e.g. e4_train_compare.py --export-onnx) works too.
# Needs JetPack (TensorRT + trtexec) and a CUDA-enabled PyTorch for the timing script.
set -euo pipefail
cd "$(dirname "$0")/.."

ONNX=${1:-runs/classify/magdalena/plant_cls_n_blocksplit_cpu/weights/best.onnx}
TAG=${2:-yolo26n}
OUT=experiments/results/e6_latency
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
SHAPES="--minShapes=images:1x3x96x320 --optShapes=images:5x3x96x320 --maxShapes=images:5x3x96x320"
mkdir -p "$OUT"

{
  echo "date: $(date -Is)"
  cat /etc/nv_tegra_release 2>/dev/null || true
  nvpmodel -q 2>/dev/null || true
  python3 -c "import tensorrt, torch; print('TensorRT', tensorrt.__version__, '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
} | tee "$OUT/${TAG}_platform.txt"

# Locked clocks give repeatable timings; report the power mode above in the paper.
sudo jetson_clocks || echo "jetson_clocks not applied: timings will vary more"

for PREC in fp16 fp32; do
  ENGINE="${ONNX%.onnx}_${PREC}.engine"
  if [ ! -f "$ENGINE" ]; then
    FLAG=""; [ "$PREC" = fp16 ] && FLAG="--fp16"
    "$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" $FLAG $SHAPES > "$OUT/${TAG}_${PREC}_trtexec.log" 2>&1
  fi
  python3 experiments/e6_benchmark_latency.py --engine "$ENGINE" --tag "${TAG}_${PREC}" --out "$OUT"
done

echo
echo "Results in $OUT. For the ROS 2 node set engine_path in ros2_ws/config/magdalena_yolo.yaml to:"
echo "  $(realpath "${ONNX%.onnx}_fp16.engine")"
