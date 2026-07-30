#!/usr/bin/env bash
# Full precision ladder on the RTX 5090 box (runs ON that box from ~/coco_journal):
# per model: [FP8 quantize if missing] -> build fp32/fp16/int8/fp8 strongly-typed
# engines (TRT 11.1) -> infer val2017 -> standard COCOeval. INT8/FP16 ONNX are
# the ones quantized once on the pilot box (quantize-once, build-per-device).
#   MODELS="yolo11n ..." nohup bash tools/coco_5090_run.sh > run.log 2>&1 &
set -u
cd ~/coco_journal
PY=~/miniconda3/envs/qtsd/bin/python
EXP=exports/coco_pilot
MET=metrics/coco_pilot
MODELS=${MODELS:-"yolo11n yolo11s yolo11m yolo11l yolo11x"}

echo "[5090] waiting for COCO download..."
while ! grep -q "COCO_DL_DONE" dl.log 2>/dev/null; do sleep 30; done
echo "[5090] COCO ready: $(ls data/coco/images/val2017 | wc -l) images"

step() { echo "[$(date +%H:%M:%S)] $*"; }

for M in $MODELS; do
  echo "================ $M ================"
  [ -f "$EXP/$M.onnx" ] || { echo "SKIP $M (no onnx yet)"; continue; }

  [ -f "$EXP/${M}_fp8.onnx" ] || { step "$M: ModelOpt FP8 quantize"; $PY tools/coco_quantize_onnx.py --onnx "$EXP/$M.onnx" --mode fp8 || { echo "FAIL fp8-quant $M"; continue; }; }

  for P in fp32 fp16 int8 fp8; do
    SRC="$EXP/${M}_${P}.onnx"; [ "$P" = "fp32" ] && SRC="$EXP/$M.onnx"
    [ -f "$SRC" ] || { echo "SKIP $M $P (no onnx)"; continue; }
    if [ ! -f "$EXP/${M}_${P}.plan" ]; then
      step "$M: build $P engine"
      $PY tools/build_engine.py --onnx "$SRC" --engine "$EXP/${M}_${P}.plan" || { echo "BUILD_FAIL $M $P"; continue; }
    fi
    PRED="$MET/pred_${M}_${P}.json"
    [ -f "$PRED" ] || { step "$M: infer $P"; $PY tools/coco_infer_trt.py --engine "$EXP/${M}_${P}.plan" --out "$PRED" || { echo "FAIL infer $M $P"; continue; }; }
    [ -f "$MET/${M}_${P}.json" ] || { step "$M: eval $P"; $PY tools/coco_eval_pilot.py --dt "$PRED" --model "$M" --precision "$P" || { echo "FAIL eval $M $P"; continue; }; }
  done
done

echo "================ summary5090 ================"
MODELS="$MODELS" $PY - <<'EOF'
import json, glob, os
rows = {}
for p in sorted(glob.glob("metrics/coco_pilot/*_*.json")):
    try: d = json.load(open(p))
    except Exception: continue
    if "model" in d and "stats" in d:
        rows.setdefault(d["model"], {})[d["precision"]] = d["stats"]
for m in os.environ.get("MODELS", "").split():
    for p in ["fp32", "fp16", "int8", "fp8"]:
        s = rows.get(m, {}).get(p)
        if s:
            print(f"{m:9} {p:5} AP={s['AP']:.4f} AP_S={s['AP_small']:.4f} AP_M={s['AP_medium']:.4f} AP_L={s['AP_large']:.4f}")
EOF
echo "done5090."
