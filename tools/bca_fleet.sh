#!/usr/bin/env bash
# Full leave-one-out BCa + 90% intervals. One process per model, 5 at a time,
# so the shared box keeps ~9 cores for everything else.
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
M=metrics/coco_pilot/bca
CNN="yolo11n yolo11s yolo11m yolo11l yolo11x yolov8n yolov8s yolov8m yolov8l yolov8x yolo26n yolo26s yolo26m yolo26l yolo26x"
run () {  # $1 model  $2 precision
  local out=$M/bca_$2_$1.json
  [ -s "$out" ] && { echo "skip $1 $2"; return; }
  $PY tools/coco_boot_diff.py --models "$1" --precision "$2" --n-boot 2000 --bca \
      --out "$out" > $M/log_$2_$1.txt 2>&1 && echo "done $1 $2 $(date +%H:%M)" || echo "FAIL $1 $2"
}
export -f run; export PY M
echo "BCA_FLEET_START $(date "+%F %T")"
# FP8 arm for all fifteen (equivalence claims), then INT8 for the YOLO11 sweep
for m in $CNN; do echo "$m fp8"; done   > $M/joblist
for m in yolo11n yolo11s yolo11m yolo11l yolo11x; do echo "$m int8"; done >> $M/joblist
xargs -a $M/joblist -P 5 -n 2 bash -c "run \$0 \$1"
echo "BCA_FLEET_DONE $(date "+%F %T")"
