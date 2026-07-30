#!/bin/bash
# Sequential training queue for the rest of README §6's detector zoo.
# Runs on the remote box itself (nohup'd), so it survives SSH disconnects.
# Waits for an existing training PID first if one is still running.
set -uo pipefail
cd /home/thuan/traffic
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "waiting for pid $WAIT_PID to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  echo "pid $WAIT_PID done, starting queue"
fi

echo "=== yolo11s ==="
yolo train data=TT100K.yaml model=yolo11s.pt imgsz=1280 epochs=100 batch=16 \
  project=runs name=yolo11s_fp32 patience=40 close_mosaic=20 device=0
echo "=== yolo11s DONE ==="

echo "=== yolov8s ==="
yolo train data=TT100K.yaml model=yolov8s.pt imgsz=1280 epochs=100 batch=16 \
  project=runs name=yolov8s_fp32 patience=40 close_mosaic=20 device=0
echo "=== yolov8s DONE ==="

echo "=== rtdetr-l ==="
yolo train data=TT100K.yaml model=rtdetr-l.pt imgsz=1280 epochs=100 batch=8 \
  project=runs name=rtdetr_fp32 patience=40 close_mosaic=20 device=0
echo "=== rtdetr-l DONE ==="

echo "ALL_TRAINING_DONE"
