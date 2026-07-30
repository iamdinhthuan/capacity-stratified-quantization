#!/bin/bash
# Capacity sweep round 2 (reviewer PHẦN A#3 / PHẦN C P1).
#
# Goal: complete the YOLO11 family sweep n/s/m/l/x so the capacity axis is
# measured WITHIN one architecture (11s vs 11m already trained; adding n/l/x
# gives 5 points over a 22x parameter range with the architecture held fixed).
# yolo26n and yolov8s stay as cross-family checks -> 7 points total.
#
# Ordered by value-per-hour, so an early abort still leaves the axis useful:
#   yolo11n  (2.6M, ~2h)  - anchors the bottom, cheap
#   yolo11x  (57M, ~10h)  - anchors the top, extends the range the most
#   yolo11l  (25M, ~6h)   - fills 20->57M, confirms monotonicity
#   yolov8m  (26M, ~6h)   - optional 2nd cross-family point at high capacity
#
# batch differs per model to fit 32GB at imgsz=1280; Ultralytics' nbs=64
# gradient accumulation normalizes the effective optimization step, and the
# already-trained models used the same convention (16 for s-class, 8 for m).
set -uo pipefail
cd /home/thuan/traffic
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd

run() {
  local name=$1 model=$2 batch=$3
  echo "=== ${name} START ==="
  yolo train data=TT100K.yaml model=${model} imgsz=1280 epochs=100 batch=${batch} \
    project=runs name=${name}_fp32 patience=40 close_mosaic=20 device=0
  echo "=== ${name} DONE ==="
}

run yolo11n yolo11n.pt 16
run yolo11x yolo11x.pt 4
run yolo11l yolo11l.pt 8
run yolov8m yolov8m.pt 8

echo "ALL_CAPACITY_TRAINING_DONE"
