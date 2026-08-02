#!/bin/bash
# Relaunch the two parts that failed: the FP32 arm of the matched-subset
# backend comparison (wrong plan name) and the entropy sweep (the quantizer
# had no calibrator flag until now).
set -uo pipefail
cd /home/thuan/coco_journal
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
E=exports/coco_pilot; M=metrics/coco_pilot

echo "=== A2. TensorRT FP32 on the same 500 images $(date '+%F %T') ==="
[ -f "$M/sub500_dfine_s_fp32.json" ] || {
  $PY tools/dfine_infer_trt.py --engine $E/dfine_s_fp32.plan \
      --out $M/pred_sub500_dfine_s_fp32.json --limit 500 &&
  $PY tools/coco_eval_pilot.py --dt $M/pred_sub500_dfine_s_fp32.json \
      --model sub500_dfine_s --precision fp32 --img-ids-from-dt; }
echo "SUBSET_FP32_DONE"

echo "=== B2. entropy calibration across the whole CNN set $(date '+%F %T') ==="
for M0 in yolov8n yolov8s yolov8m yolov8l yolov8x \
          yolo26n yolo26s yolo26m yolo26l yolo26x \
          yolo11s yolo11l yolo11x; do
  [ -f "$M/${M0}_int8ent.json" ] && { echo "  have $M0"; continue; }
  [ -f "$E/${M0}_int8ent.onnx" ] || {
    $PY tools/coco_quantize_onnx.py --onnx $E/${M0}.onnx --mode int8 --calib entropy \
      || { echo "QUANT_FAIL $M0"; continue; }
    mv $E/${M0}_int8.onnx $E/${M0}_int8ent.onnx 2>/dev/null; }
  [ -f "$E/${M0}_int8ent.plan" ] || \
    $PY tools/build_engine.py --onnx $E/${M0}_int8ent.onnx --engine $E/${M0}_int8ent.plan \
      || { echo "BUILD_FAIL $M0"; continue; }
  $PY tools/coco_infer_trt.py --engine $E/${M0}_int8ent.plan --out $M/pred_${M0}_int8ent.json \
    || { echo "INFER_FAIL $M0"; continue; }
  $PY tools/coco_eval_pilot.py --dt $M/pred_${M0}_int8ent.json --model "$M0" --precision int8ent \
    || echo "EVAL_FAIL $M0"
  echo "  done $M0 $(date '+%H:%M')"
done
echo "ENTROPY_SWEEP2_DONE $(date '+%F %T')"
