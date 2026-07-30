#!/usr/bin/env bash
# Overnight Orin pipeline (runs on the 4090 box):
#   1. wait for the COCO pilot queue to finish (frees CPU for calibration)
#   2. generate hp32-flavor INT8 ONNX for yolo11 s/m/l/x
#      (high_precision_dtype=fp32 — required by the Jetson TRT 10.3 parser;
#       verified: default fp16 flavor asserts in weightsPtr.h on 10.3)
#   3. rsync all ONNX to the Orin
#   4. build strongly-typed engines on-device (fp32 / fp16 / int8-hp32)
#   5. run orin_bench.sh over every engine, pull results back
set -u
cd "$(dirname "$0")/.."
VENV=/data_nvme/paper/.venv_pilot/bin/python
# Jetson device. Set ORIN_HOST=user@address for your own board.
ORIN=${ORIN_HOST:?set ORIN_HOST=user@address for the Jetson board}
EXP=exports/coco_pilot

echo "[orin_queue] waiting for pilot queue to finish..."
while ! grep -q "^done\.$" runs_coco_pilot.log 2>/dev/null; do sleep 60; done
echo "[orin_queue] pilot done at $(date +%H:%M:%S)"

for M in yolo11s yolo11m yolo11l yolo11x; do
  if [ ! -f "$EXP/${M}_int8_hp32.onnx" ]; then
    echo "[orin_queue] hp32 quantize $M"
    $VENV - <<EOF || { echo "FAIL hp32 $M"; continue; }
import glob, os, sys, numpy as np, onnx
sys.path.insert(0, 'tools')
from coco_common import COCO_CALIB_DIR, preprocess
from modelopt.onnx.quantization import quantize as mq
files = sorted(glob.glob(os.path.join(COCO_CALIB_DIR, '*.jpg')))[:512]
calib = np.stack([preprocess(f, 640)[0][0] for f in files]).astype(np.float32)
p = '$EXP/$M.onnx'
inp = onnx.load(p, load_external_data=False).graph.input[0].name
mq(p, quantize_mode='int8', calibration_data={inp: calib}, calibration_method='max',
   calibration_eps=['cpu'], output_path='$EXP/${M}_int8_hp32.onnx',
   op_types_to_exclude=['Sigmoid'], high_precision_dtype='fp32')
EOF
  fi
done

echo "[orin_queue] rsync onnx -> orin"
rsync -q -e "ssh -o BatchMode=yes" \
  $EXP/yolo11{s,m,l,x}.onnx $EXP/yolo11{s,m,l,x}_fp16.onnx $EXP/yolo11{s,m,l,x}_int8_hp32.onnx \
  $ORIN:~/qtsd_edge/onnx/ || echo "[orin_queue] rsync had errors"

echo "[orin_queue] building engines on orin (long)..."
ssh -o BatchMode=yes $ORIN '
T=/usr/src/tensorrt/bin/trtexec
cd ~/qtsd_edge
for M in yolo11s yolo11m yolo11l yolo11x; do
  for P in fp32 fp16 int8; do
    case $P in
      fp32) SRC=onnx/$M.onnx;;
      fp16) SRC=onnx/${M}_fp16.onnx;;
      int8) SRC=onnx/${M}_int8_hp32.onnx;;
    esac
    [ -f "$SRC" ] || { echo "SKIP $M $P (no onnx)"; continue; }
    if [ ! -f engines/${M}_${P}.plan ]; then
      echo "[orin] build $M $P $(date +%H:%M:%S)"
      $T --onnx=$SRC --stronglyTyped --saveEngine=engines/${M}_${P}.plan --skipInference \
        > logs/build_${M}_${P}.log 2>&1 && echo "BUILD_OK $M $P" || { echo "BUILD_FAIL $M $P"; grep -E "\[E\]" logs/build_${M}_${P}.log | head -2; }
    fi
  done
done
bash ~/qtsd_edge/orin_bench.sh ~/qtsd_edge/engines'

mkdir -p metrics/orin
scp -q -o BatchMode=yes $ORIN:~/qtsd_edge/bench_results.jsonl metrics/orin/bench_results.jsonl && echo "[orin_queue] results pulled -> metrics/orin/bench_results.jsonl"
echo "[orin_queue] ALL DONE $(date +%H:%M:%S)"
