#!/usr/bin/env bash
# Confirmatory chain (runs on the 4090 box):
#   1. wait for the local YOLOv8 queue to finish
#   2. rsync the v8 ONNX set to the 5090 box
#   3. on the 5090: v8 full ladder (fp32/fp16/int8/fp8, canonical device),
#      then YOLO26: export ONNX (qtsd has ultralytics) + fp16/int8 quantize
#      + full ladder (e2e NMS-free head handled by the updated decoder).
set -u
cd "$(dirname "$0")/.."
# Desktop build/eval host. Set GPU_HOST=user@address for your own machine.
T5090=${GPU_HOST:?set GPU_HOST=user@address for the desktop GPU box}

echo "[chain] waiting for local v8 queue..."
while ! grep -q "^done\.$" runs_coco_v8.log 2>/dev/null; do sleep 60; done
echo "[chain] v8 queue done at $(date +%H:%M:%S)"

FILES=""
for M in n s m l x; do for S in "" "_fp16" "_int8"; do
  F="exports/coco_pilot/yolov8$M$S.onnx"; [ -f "$F" ] && FILES="$FILES $F"
done; done
rsync -q -e "ssh -o BatchMode=yes" $FILES $T5090:~/coco_journal/exports/coco_pilot/ && echo "[chain] v8 onnx synced"

ssh -o BatchMode=yes $T5090 'cd ~/coco_journal && nohup bash -c "
MODELS=\"yolov8n yolov8s yolov8m yolov8l yolov8x\" bash tools/coco_5090_run.sh >> run_v8.log 2>&1
PY=~/miniconda3/envs/qtsd/bin/python
mkdir -p exports/coco_pilot && cd exports/coco_pilot
for M in n s m l x; do
  [ -f yolo26\$M.onnx ] || \$PY -c \"from ultralytics import YOLO; YOLO(\x27yolo26\$M.pt\x27).export(format=\x27onnx\x27, imgsz=640, batch=1, dynamic=False, simplify=True, opset=17)\" >> ../../run_26.log 2>&1
done
cd ~/coco_journal
for M in n s m l x; do
  [ -f exports/coco_pilot/yolo26\${M}_fp16.onnx ] || \$PY tools/coco_quantize_onnx.py --onnx exports/coco_pilot/yolo26\$M.onnx --mode fp16 >> run_26.log 2>&1
  [ -f exports/coco_pilot/yolo26\${M}_int8.onnx ] || \$PY tools/coco_quantize_onnx.py --onnx exports/coco_pilot/yolo26\$M.onnx --mode int8 >> run_26.log 2>&1
done
MODELS=\"yolo26n yolo26s yolo26m yolo26l yolo26x\" bash tools/coco_5090_run.sh >> run_26.log 2>&1
echo CHAIN_ALL_DONE >> run_26.log
" > /dev/null 2>&1 & echo "[chain] remote chain launched"'
echo "[chain] done handing off."
