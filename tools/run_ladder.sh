#!/bin/bash
# Full precision ladder for ONE detector, end to end (README §7-§8).
# Run only when the GPU is free (check `nvidia-smi` first) — every step
# after export needs the GPU, and several run back-to-back.
#
# READY end-to-end: fp32, fp16, int8_ptq (native Ultralytics export+eval path).
# EXPERIMENTAL / blocked: int8_qat, fp8, fp4 — these export through the raw
# modelopt torch.onnx.export path (tools/quantize_modelopt.py) instead of
# Ultralytics' own exporter, so the resulting engine is NOT loadable by
# tools/infer.py. Eval for those three goes through tools/infer_trt_raw.py,
# whose decode_output() is an intentional NotImplementedError until someone
# inspects a real engine's raw output shape (see that file's docstring).
# Don't trust fp8/fp4/int8_qat numbers until that's filled in.
#
# Usage: tools/run_ladder.sh <model_name> <best_pt_path> [imgsz]
set -euo pipefail

MODEL=$1
BEST_PT=$2
IMGSZ=${3:-1280}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$ROOT/data/TT100K"
TEST_IMAGES="$DATA_ROOT/images/test"
GT="$ROOT/metrics/gt_test.json"

cd "$ROOT"

if [ ! -f "$GT" ]; then
  python tools/gt_to_coco.py --data-root "$DATA_ROOT" --split test --out "$GT"
fi
if [ ! -d "$ROOT/data/calib/images" ]; then
  python tools/build_calib.py --data-root "$DATA_ROOT" --out-dir "$ROOT/data/calib" --n 300
fi

eval_ultralytics() {
  local precision=$1 weights=$2
  python tools/infer.py --model "$weights" --images-dir "$TEST_IMAGES" --imgsz "$IMGSZ" \
      --out "metrics/pred_${MODEL}_${precision}.json"
  python tools/eval_stratified.py --gt "$GT" --dt "metrics/pred_${MODEL}_${precision}.json" \
      --model "$MODEL" --precision "$precision" --out "metrics/${MODEL}_${precision}.json"
  python tools/measure_system.py --weights "$weights" --imgsz "$IMGSZ" \
      --metrics-json "metrics/${MODEL}_${precision}.json"
}

eval_raw_engine() {
  local precision=$1 engine=$2
  python tools/infer_trt_raw.py --engine "$engine" --images-dir "$TEST_IMAGES" --imgsz "$IMGSZ" \
      --out "metrics/pred_${MODEL}_${precision}.json"
  python tools/eval_stratified.py --gt "$GT" --dt "metrics/pred_${MODEL}_${precision}.json" \
      --model "$MODEL" --precision "$precision" --out "metrics/${MODEL}_${precision}.json"
}

echo "=== FP32 (baseline, already trained) ==="
eval_ultralytics fp32 "$BEST_PT"

echo "=== FP16 ==="
python tools/export_engine.py --weights "$BEST_PT" --mode fp16 --imgsz "$IMGSZ"
eval_ultralytics fp16 "${BEST_PT%.pt}.engine"

echo "=== INT8-PTQ ==="
python tools/export_engine.py --weights "$BEST_PT" --mode int8 --imgsz "$IMGSZ" \
    --calib-yaml configs/tt100k_calib.yaml
eval_ultralytics int8_ptq "${BEST_PT%.pt}.engine"

echo "=== INT8-QAT (experimental — see header) ==="
python tools/quantize_modelopt.py --weights "$BEST_PT" --mode int8 \
    --calib-dir data/calib/images --imgsz "$IMGSZ" --finetune-epochs 8 \
    --data configs/tt100k.yaml --out "exports/${MODEL}_int8qat.onnx" || echo "SKIPPED: quantize_modelopt failed"
python tools/build_engine.py --onnx "exports/${MODEL}_int8qat.onnx" --engine "exports/${MODEL}_int8qat.plan" || echo "SKIPPED: build_engine failed"
eval_raw_engine int8_qat "exports/${MODEL}_int8qat.plan" || echo "SKIPPED: decode_output not implemented yet"

echo "=== FP8 (E4M3, experimental — see header) ==="
python tools/quantize_modelopt.py --weights "$BEST_PT" --mode fp8 \
    --calib-dir data/calib/images --imgsz "$IMGSZ" --out "exports/${MODEL}_fp8.onnx" || echo "SKIPPED: quantize_modelopt failed"
python tools/build_engine.py --onnx "exports/${MODEL}_fp8.onnx" --engine "exports/${MODEL}_fp8.plan" || echo "SKIPPED: build_engine failed"
eval_raw_engine fp8 "exports/${MODEL}_fp8.plan" || echo "SKIPPED: decode_output not implemented yet"

echo "=== FP4 (NVFP4, experimental — see header) ==="
python tools/quantize_modelopt.py --weights "$BEST_PT" --mode nvfp4 \
    --calib-dir data/calib/images --imgsz "$IMGSZ" --out "exports/${MODEL}_fp4.onnx" || echo "SKIPPED: quantize_modelopt failed"
python tools/build_engine.py --onnx "exports/${MODEL}_fp4.onnx" --engine "exports/${MODEL}_fp4.plan" || echo "SKIPPED: build_engine failed"
eval_raw_engine fp4 "exports/${MODEL}_fp4.plan" || echo "SKIPPED: decode_output not implemented yet"

echo "=== Tables (uses whichever precisions succeeded above) ==="
python tools/make_tables.py --metrics-dir metrics --out-dir tables
