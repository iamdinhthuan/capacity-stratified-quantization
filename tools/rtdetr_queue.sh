#!/bin/bash
# Is the DETR-class collapse a property of the class, or of D-FINE?
#
# The paper now leans on that distinction: the pre-registered contrast failed
# because the transformer family shares no regime with the convolutional ones,
# and that explanation appears in the abstract, the contributions and the
# conclusion. It rests on one family. A second DETR-class detector run through
# the identical recipe either makes it a class property or narrows the claim to
# D-FINE alone.
#
# This can go against the paper. If RT-DETR quantizes cleanly, the architectural
# boundary shrinks to a single family and several sentences have to be rewritten.
# That is the point of running it.
#
#   setsid nohup bash tools/rtdetr_queue.sh > run_rtdetr.log 2>&1 < /dev/null &
set -uo pipefail
cd /home/thuan/coco_journal
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
E=exports/coco_pilot
M=metrics/coco_pilot
mkdir -p "$E" "$M"

# ---- wait for the seed queue to finish, then for the GPU ------------------
# Two conditions, not one. GPU-idle alone would fire in the gap between a
# training run and its ladder, and two jobs building engines at once is how
# you get an out-of-memory failure attributed to the wrong experiment. The
# marker is read from a file rather than matched against a process name,
# because a name match would also match this script's own command line.
SEEDLOG=/home/thuan/traffic/run_seeds.log
echo "[rtdetr] waiting for the seed queue to finish..."
while ! grep -q "SEED_QUEUE_ALL_DONE" "$SEEDLOG" 2>/dev/null; do sleep 60; done
echo "[rtdetr] seed queue done at $(date '+%F %T')"
# Wait for memory headroom, not for an empty GPU. This job measures accuracy,
# which is deterministic under contention; only the latency tables need an
# idle device, and none are produced here. The box is shared, so requiring
# zero compute apps would wait forever.
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge 4000 ] && break
  echo "[rtdetr] only ${free} MiB free, waiting"; sleep 120
done
echo "[rtdetr] ${free} MiB free at $(date '+%F %T'); starting"

MODELS="rtdetr-l rtdetr-x"

# ---- export -----------------------------------------------------------------
echo "=== EXPORT $(date '+%F %T') ==="
$PY - <<'EOF'
from ultralytics import RTDETR
import shutil, os
for name in ("rtdetr-l", "rtdetr-x"):
    dst = f"exports/coco_pilot/{name.replace('-','_')}.onnx"
    if os.path.exists(dst):
        print("  have", dst, flush=True); continue
    m = RTDETR(f"{name}.pt")
    p = m.export(format="onnx", imgsz=640, batch=1, opset=17, simplify=True, dynamic=False)
    shutil.copy(p, dst)
    print("  ->", dst, flush=True)
EOF

# ---- reference AP from the framework itself, for the fidelity gate ----------
echo "=== PYTORCH REFERENCE $(date '+%F %T') ==="
$PY - <<'EOF'
import json, os
from ultralytics import RTDETR
out = {}
for name in ("rtdetr-l", "rtdetr-x"):
    r = RTDETR(f"{name}.pt").val(data="coco.yaml", imgsz=640, batch=1, conf=0.001,
                                 iou=0.7, save_json=False, verbose=False)
    out[name] = {"AP": float(r.box.map), "AP50": float(r.box.map50),
                 "AP_small": float(r.box.maps[0]) if hasattr(r.box, "maps") else None}
    print(f"  {name}: torch AP = {out[name]['AP']:.4f}", flush=True)
json.dump(out, open("metrics/coco_pilot/rtdetr_torch_reference.json", "w"), indent=1)
EOF

# ---- decoder + FP32 fidelity gate -------------------------------------------
# Nothing quantized is evaluated until the FP32 engine reproduces the framework
# number. This is the same gate that caught the D-FINE GridSample defect.
echo "=== FP32 GATE $(date '+%F %T') ==="
for M0 in rtdetr_l rtdetr_x; do
  [ -f "$E/${M0}.plan" ] || $PY tools/build_engine.py --onnx "$E/${M0}.onnx" --engine "$E/${M0}.plan" || echo "BUILD_FAIL ${M0}_fp32"
done
$PY tools/rtdetr_gate.py --models rtdetr_l rtdetr_x --tol 0.01 --probe 500 || {
  echo "RTDETR_GATE_FAILED — decode or export is wrong; no quantized number will be produced"
  echo "RTDETR_ABORTED $(date '+%F %T')"; exit 1; }

# ---- the ladder, identical recipe to every other family ---------------------
echo "=== LADDER $(date '+%F %T') ==="
for M0 in rtdetr_l rtdetr_x; do
  for MODE in fp16 int8 fp8; do
    [ -f "$E/${M0}_${MODE}.onnx" ] || \
      $PY tools/coco_quantize_onnx.py --onnx "$E/${M0}.onnx" --mode "$MODE" \
        || { echo "QUANT_FAIL ${M0}_${MODE}"; continue; }
    [ -f "$E/${M0}_${MODE}.plan" ] || \
      $PY tools/build_engine.py --onnx "$E/${M0}_${MODE}.onnx" --engine "$E/${M0}_${MODE}.plan" \
        || { echo "BUILD_FAIL ${M0}_${MODE}"; continue; }
  done
  for MODE in fp32 fp16 int8 fp8; do
    PLAN="$E/${M0}.plan"; [ "$MODE" = fp32 ] || PLAN="$E/${M0}_${MODE}.plan"
    [ -f "$PLAN" ] || { echo "SKIP ${M0}_${MODE} (no engine)"; continue; }
    [ -f "$M/${M0}_${MODE}.json" ] && continue
    RMODE=$($PY -c "import json;print(json.load(open('metrics/coco_pilot/rtdetr_gate.json'))['chosen_mode']['${M0}'])")
    $PY tools/coco_infer_trt.py --engine "$PLAN" --decoder rtdetr --rtdetr-mode "$RMODE" \
        --out "$M/pred_${M0}_${MODE}.json" || { echo "INFER_FAIL ${M0}_${MODE}"; continue; }
    $PY tools/coco_eval_pilot.py --dt "$M/pred_${M0}_${MODE}.json" \
        --model "$M0" --precision "$MODE" || echo "EVAL_FAIL ${M0}_${MODE}"
  done
done

echo "=== SUMMARY vs D-FINE $(date '+%F %T') ==="
$PY - <<'EOF'
import json, os
M = "metrics/coco_pilot"; M5 = "metrics/coco_5090"
def load(d, m, p):
    f = os.path.join(d, f"{m}_{p}.json")
    return json.load(open(f))["stats"] if os.path.exists(f) else None
print(f"{'model':10s} {'FP32':>7s} {'INT8':>7s} {'FP8':>7s} {'INT8 loss':>10s} {'FP8 loss':>9s}")
for d, m in [(M, "rtdetr_l"), (M, "rtdetr_x"), (M5, "dfine_l"), (M5, "dfine_x")]:
    r = load(d, m, "fp32")
    if not r: print(f"{m:10s} pending"); continue
    q8, qf = load(d, m, "int8"), load(d, m, "fp8")
    print(f"{m:10s} {r['AP']:7.4f} "
          f"{q8['AP'] if q8 else float('nan'):7.4f} {qf['AP'] if qf else float('nan'):7.4f} "
          f"{(r['AP']-q8['AP'])*100 if q8 else float('nan'):9.2f}pt "
          f"{(r['AP']-qf['AP'])*100 if qf else float('nan'):8.2f}pt")
print()
print("Read: if RT-DETR loses tens of points like D-FINE, the boundary is the")
print("architecture class. If it loses a point or two, the boundary is D-FINE")
print("alone and the manuscript overstates the claim.")
EOF
echo "RTDETR_ALL_DONE $(date '+%F %T')"
