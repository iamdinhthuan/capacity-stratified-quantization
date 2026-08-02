#!/usr/bin/env bash
# The two YOLO11 rungs the entropy sweep skipped, through the identical pipeline.
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
E=exports/coco_pilot; M=metrics/coco_pilot
echo "ENTROPY_FINISH_START $(date "+%F %T")"
for M0 in yolo11n yolo11m; do
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
  echo "  done $M0 $(date "+%H:%M")"
done
echo "ENTROPY_FINISH_DONE $(date "+%F %T")"
