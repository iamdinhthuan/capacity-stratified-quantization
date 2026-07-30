#!/bin/bash
# Full precision ladder + stratified eval for ONE trained model, for expanding
# the capacity sweep. Reuses the validated tools. Usage:
#   tools/model_ladder.sh <model_name>    (expects runs/detect/runs/<name>_fp32/weights/best.pt)
set -uo pipefail
cd /home/thuan/traffic
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd

M=$1
BEST=runs/detect/runs/${M}_fp32/weights/best.pt
IMG=data/TT100K/images/test
GT=metrics/gt_test.json
IMGSZ=1280

echo "=== ${M}: FP32 ==="
python tools/infer.py --model "$BEST" --images-dir "$IMG" --imgsz $IMGSZ --out metrics/pred_${M}_fp32.json
python tools/eval_stratified.py --gt "$GT" --dt metrics/pred_${M}_fp32.json --model "$M" --precision fp32 --out metrics/${M}_fp32.json

echo "=== ${M}: FP16 ==="
python tools/export_engine.py --weights "$BEST" --mode fp16 --imgsz $IMGSZ
python tools/infer.py --model "${BEST%.pt}.fp16.engine" --images-dir "$IMG" --imgsz $IMGSZ --out metrics/pred_${M}_fp16.json
python tools/eval_stratified.py --gt "$GT" --dt metrics/pred_${M}_fp16.json --model "$M" --precision fp16 --out metrics/${M}_fp16.json

echo "=== ${M}: INT8-PTQ ==="
python tools/export_engine.py --weights "$BEST" --mode int8 --imgsz $IMGSZ --calib-yaml configs/tt100k_calib.yaml
python tools/infer.py --model "${BEST%.pt}.int8.engine" --images-dir "$IMG" --imgsz $IMGSZ --out metrics/pred_${M}_int8_ptq.json
python tools/eval_stratified.py --gt "$GT" --dt metrics/pred_${M}_int8_ptq.json --model "$M" --precision int8_ptq --out metrics/${M}_int8_ptq.json

echo "=== ${M}: FP8 ==="
python tools/export_engine.py --weights "$BEST" --mode onnx --imgsz $IMGSZ
python3 -m modelopt.onnx.quantization --onnx_path "${BEST%.pt}.onnx" --quantize_mode fp8 \
    --calibration_data_path data/calib/calib_inputs.npy --output_path exports/${M}_fp8_cli.onnx
python tools/build_engine.py --onnx exports/${M}_fp8_cli.onnx --engine exports/${M}_fp8.plan
python tools/infer_trt_raw.py --engine exports/${M}_fp8.plan --images-dir "$IMG" --imgsz $IMGSZ --conf 0.001 --out metrics/pred_${M}_fp8.json
python tools/eval_stratified.py --gt "$GT" --dt metrics/pred_${M}_fp8.json --model "$M" --precision fp8 --out metrics/${M}_fp8.json

echo "LADDER_DONE_${M}"
python3 -c "
import json
f=json.load(open('metrics/${M}_fp32.json')); i=json.load(open('metrics/${M}_int8_ptq.json'))
dS=f['height_bin_ap']['S']['mAP50-95']-i['height_bin_ap']['S']['mAP50-95']
dXL=f['height_bin_ap']['XL']['mAP50-95']-i['height_bin_ap']['XL']['mAP50-95']
print(f'${M}: FP32={f[\"overall\"][\"mAP50-95\"]:.3f} INT8={i[\"overall\"][\"mAP50-95\"]:.3f} dS={dS:.4f} dXL={dXL:.4f} SUR={dS/dXL:.2f}')
"
