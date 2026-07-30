#!/usr/bin/env bash
# Referee-driven experiments on the 5090 (run from ~/coco_journal).
#  M5a: D-FINE under ENTROPY calibration — the CNN ablation showed max->entropy
#       halves INT8 loss, and max-calibration is the known failure mode for
#       transformer activation outliers, so the "DETR collapses under PTQ"
#       claim must be tested with a second calibrator before it can stand.
#  M7 : latency re-measured with RANDOMISED rung order and 3 repeats, so the
#       "FP8 is the fastest rung" claim is not confounded with measurement order.
set -u
cd ~/coco_journal
PY=~/miniconda3/envs/qtsd/bin/python
EXP=exports/coco_pilot
MET=metrics/coco_pilot

echo "=== M5a: D-FINE entropy calibration ==="
for M in n s m l x; do
  BASE="$EXP/dfine_${M}_explicit.onnx"
  OUT="$EXP/dfine_${M}_explicit_int8ent.onnx"
  [ -f "$BASE" ] || { echo "SKIP dfine_$M"; continue; }
  if [ ! -f "$OUT" ]; then
    $PY - <<PYEOF || { echo "FAIL quant dfine_$M"; continue; }
import glob, os, sys, numpy as np, onnx
sys.path.insert(0, "tools")
from coco_common import COCO_CALIB_DIR
from PIL import Image
from modelopt.onnx.quantization import quantize as mq
files = sorted(glob.glob(os.path.join(COCO_CALIB_DIR, "*.jpg")))[:512]
imgs, sizes = [], []
for f in files:
    im = Image.open(f).convert("RGB"); w0, h0 = im.size
    imgs.append(np.asarray(im.resize((640, 640), Image.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255.0)
    sizes.append([w0, h0])
X = np.ascontiguousarray(np.stack(imgs)); S = np.asarray(sizes, dtype=np.int64)
g = onnx.load("$BASE", load_external_data=False).graph
ins = [i.name for i in g.input]
mq("$BASE", quantize_mode="int8", calibration_data={ins[0]: X, ins[1]: S},
   calibration_method="entropy", calibration_eps=["cpu"],
   output_path="$OUT", op_types_to_exclude=["Sigmoid"])
PYEOF
  fi
  P="$EXP/dfine_${M}_int8ent.plan"
  [ -f "$P" ] || $PY tools/build_engine.py --onnx "$OUT" --engine "$P" || continue
  PRED="$MET/pred_dfine_${M}_int8ent.json"
  [ -f "$PRED" ] || $PY tools/dfine_infer_trt.py --engine "$P" --out "$PRED" || continue
  [ -f "$MET/dfine_${M}_int8ent.json" ] || $PY tools/coco_eval_pilot.py --dt "$PRED" --model "dfine_$M" --precision int8ent > /dev/null
  $PY - <<PYEOF
import json
r=json.load(open("$MET/dfine_${M}_fp32.json"))["stats"]["AP"]
m=json.load(open("$MET/dfine_${M}_int8.json"))["stats"]["AP"]
e=json.load(open("$MET/dfine_${M}_int8ent.json"))["stats"]["AP"]
print(f"ENTROPY dfine_$M: fp32={r:.4f} int8-max={m:.4f} int8-entropy={e:.4f}")
PYEOF
done
echo "DFINE_ENTROPY_DONE"

echo "=== M7: randomised-order repeated latency ==="
$PY - <<'PYEOF'
import json, os, random, subprocess
PY = os.path.expanduser("~/miniconda3/envs/qtsd/bin/python")
models = ["yolo11n", "yolo11m", "yolo11x", "yolov8n", "yolov8m", "yolov8x",
          "yolo26n", "yolo26m", "yolo26x"]
jobs = [(m, p) for m in models for p in ("fp16", "int8", "fp8")]
out = "metrics/coco_pilot/latency640_v2.jsonl"
open(out, "w").close()
rng = random.Random(0)
for rep in range(3):
    order = jobs[:]
    rng.shuffle(order)                      # randomised rung order per repeat
    for m, p in order:
        eng = f"exports/coco_pilot/{m}_{p}.plan"
        if not os.path.exists(eng):
            continue
        r = subprocess.run([PY, "tools/measure_trt.py", "--engine", eng, "--imgsz", "640",
                            "--n-warmup", "100", "--n-iters", "500"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("FAIL", m, p); continue
        d = json.loads(r.stdout)
        d.update(model=m, precision=p, repeat=rep)
        with open(out, "a") as f:
            f.write(json.dumps(d) + "\n")
    print(f"repeat {rep} done", flush=True)
print("->", out)
PYEOF
echo "LATENCY_V2_DONE"
