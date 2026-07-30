#!/usr/bin/env bash
# COCO pilot queue (README_journal.md v2 §3.4): YOLO11 n/s/m/l/x
# ladder FP32/FP16/INT8 -> predictions -> COCOeval, all artifacts cached
# (a step is skipped when its output already exists, so re-runs resume).
#
#   bash tools/coco_pilot_run.sh            # full run
#   MODELS="yolo11n" LIMIT=200 bash ...     # smoke test
set -u
cd "$(dirname "$0")/.."

PY310=/home/huy/miniconda3/envs/py310/bin/python
VENV=/data_nvme/paper/.venv_pilot/bin/python
EXP=exports/coco_pilot
MET=metrics/coco_pilot
MODELS=${MODELS:-"yolo11n yolo11s yolo11m yolo11l yolo11x"}
LIMIT=${LIMIT:-0}
EVAL_FLAG=""
[ "$LIMIT" != "0" ] && EVAL_FLAG="--img-ids-from-dt"

mkdir -p "$EXP" "$MET"

step() { echo "[$(date +%H:%M:%S)] $*"; }

for M in $MODELS; do
  echo "================ $M ================"

  # 1. weights + FP32 ONNX (py310: ultralytics + torch)
  if [ ! -f "$EXP/$M.onnx" ]; then
    step "$M: export FP32 ONNX"
    (cd "$EXP" && $PY310 -c "
from ultralytics import YOLO
YOLO('$M.pt').export(format='onnx', imgsz=640, batch=1, dynamic=False, simplify=True, opset=17)
") || { echo "FAIL onnx $M"; continue; }
  fi

  # 2. FP16 AutoCast + INT8 ModelOpt ONNX (venv, CPU)
  [ -f "$EXP/${M}_fp16.onnx" ] || { step "$M: AutoCast FP16"; $VENV tools/coco_quantize_onnx.py --onnx "$EXP/$M.onnx" --mode fp16 || { echo "FAIL fp16 $M"; continue; }; }
  [ -f "$EXP/${M}_int8.onnx" ] || { step "$M: ModelOpt INT8 (CPU calib 512)"; $VENV tools/coco_quantize_onnx.py --onnx "$EXP/$M.onnx" --mode int8 || { echo "FAIL int8 $M"; continue; }; }

  # 3. strongly-typed engines (venv, GPU)
  for P in fp32 fp16 int8; do
    SRC="$EXP/${M}_${P}.onnx"; [ "$P" = "fp32" ] && SRC="$EXP/$M.onnx"
    [ -f "$EXP/${M}_${P}.plan" ] || { step "$M: build $P engine"; $VENV tools/build_engine.py --onnx "$SRC" --engine "$EXP/${M}_${P}.plan" || { echo "FAIL build $P $M"; continue 2; }; }
  done

  # 4. predictions + eval
  for P in fp32 fp16 int8; do
    PRED="$MET/pred_${M}_${P}.json"
    [ -f "$PRED" ] || { step "$M: infer $P"; $VENV tools/coco_infer_trt.py --engine "$EXP/${M}_${P}.plan" --out "$PRED" --limit "$LIMIT" || { echo "FAIL infer $P $M"; continue 2; }; }
    [ -f "$MET/${M}_${P}.json" ] || { step "$M: eval $P"; $VENV tools/coco_eval_pilot.py --dt "$PRED" --model "$M" --precision "$P" $EVAL_FLAG || { echo "FAIL eval $P $M"; continue 2; }; }
  done
done

echo "================ summary ================"
MODELS="$MODELS" $VENV - <<'EOF'
import json, glob, os
rows = {}
for p in sorted(glob.glob("metrics/coco_pilot/*_*.json")):
    try: d = json.load(open(p))
    except Exception: continue
    if "model" in d and "stats" in d:
        rows.setdefault(d["model"], {})[d["precision"]] = d["stats"]
hdr = f"{'model':9} {'prec':5} {'AP':>6} {'AP50':>6} {'AP_S':>6} {'AP_M':>6} {'AP_L':>6}"
print(hdr); print("-" * len(hdr))
for m in os.environ.get("MODELS", "").split():
    for p in ["fp32", "fp16", "int8"]:
        s = rows.get(m, {}).get(p)
        if s:
            print(f"{m:9} {p:5} {s['AP']:6.4f} {s['AP50']:6.4f} {s['AP_small']:6.4f} {s['AP_medium']:6.4f} {s['AP_large']:6.4f}")
EOF
echo "done."
