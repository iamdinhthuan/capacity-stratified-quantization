#!/usr/bin/env bash
# D-FINE full pipeline on the 5090 box (run ON that box from ~/coco_journal):
# per model: build fp32+fp16 strongly-typed engines from the explicit-deform
# ONNX -> infer val2017 -> eval (parity matrix), then ModelOpt INT8+FP8
# quantize (two-input calibration) -> build -> infer -> eval (confirmatory).
#   setsid nohup bash tools/dfine_5090_run.sh > run_dfine.log 2>&1 < /dev/null &
set -u
cd ~/coco_journal
PY=~/miniconda3/envs/qtsd/bin/python
EXP=exports/coco_pilot
MET=metrics/coco_pilot
MODELS=${MODELS:-"n s m l x"}

step() { echo "[$(date +%H:%M:%S)] $*"; }

for M in $MODELS; do
  echo "================ dfine_$M ================"
  BASE="$EXP/dfine_${M}_explicit.onnx"
  [ -f "$BASE" ] || { echo "SKIP dfine_$M (no onnx)"; continue; }

  for Q in int8 fp8; do
    [ -f "$EXP/dfine_${M}_explicit_${Q}.onnx" ] || { step "dfine_$M: quantize $Q"; $PY tools/dfine_quantize.py --onnx "$BASE" --mode $Q || echo "FAIL quant $Q dfine_$M"; }
  done

  for P in fp32 fp16 int8 fp8; do
    case $P in
      fp32) SRC="$BASE";;
      fp16) SRC="$EXP/dfine_${M}_explicit_fp16.onnx";;
      *)    SRC="$EXP/dfine_${M}_explicit_${P}.onnx";;
    esac
    [ -f "$SRC" ] || { echo "SKIP dfine_$M $P (no onnx)"; continue; }
    if [ ! -f "$EXP/dfine_${M}_${P}.plan" ]; then
      step "dfine_$M: build $P"
      $PY tools/build_engine.py --onnx "$SRC" --engine "$EXP/dfine_${M}_${P}.plan" || { echo "BUILD_FAIL dfine_$M $P"; continue; }
    fi
    PRED="$MET/pred_dfine_${M}_${P}.json"
    [ -f "$PRED" ] || { step "dfine_$M: infer $P"; $PY tools/dfine_infer_trt.py --engine "$EXP/dfine_${M}_${P}.plan" --out "$PRED" || { echo "FAIL infer dfine_$M $P"; continue; }; }
    [ -f "$MET/dfine_${M}_${P}.json" ] || { step "dfine_$M: eval $P"; $PY tools/coco_eval_pilot.py --dt "$PRED" --model "dfine_$M" --precision "$P" || echo "FAIL eval dfine_$M $P"; }
  done
done

echo "================ dfine summary ================"
$PY - <<'EOF'
import json, glob
for p in sorted(glob.glob("metrics/coco_pilot/dfine_*_*.json")):
    try: d = json.load(open(p))
    except Exception: continue
    if "stats" in d:
        s = d["stats"]
        print(f"{d['model']:9} {d['precision']:5} AP={s['AP']:.4f} AP_S={s['AP_small']:.4f} AP_L={s['AP_large']:.4f}")
EOF
echo "DFINE_ALL_DONE"
