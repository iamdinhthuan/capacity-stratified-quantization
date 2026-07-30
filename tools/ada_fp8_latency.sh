#!/bin/bash
# Does the FP8 speed advantage survive a hardware generation?
#
# Every latency number behind the paper's format recommendation comes from one
# Blackwell GPU. Grey literature reports FP8 slower than FP16 on the previous
# (Ada) generation, and the paper currently cites that rather than testing it.
# The local RTX 4090 is SM89 with FP8 tensor cores and TensorRT 10.16 compiles
# the same strongly-typed graphs there (verified), so the question is directly
# answerable with the artifacts already built.
#
# Same nine models and same protocol as the 5090 table: three repeats in
# randomised rung order, 100 warm-up + 500 timed iterations, so an ordering
# effect cannot masquerade as a format effect.
set -uo pipefail
cd /data_nvme/paper
PY=/data_nvme/paper/.venv_pilot/bin/python
W=/tmp/claude-1000/-data-nvme-paper/4fb5f984-9af2-48bc-ac1a-4fd20496f557/scratchpad/ada
mkdir -p "$W" metrics/ada

MODELS="yolo11n yolo11m yolo11x yolov8n yolov8m yolov8x yolo26n yolo26m yolo26x"

echo "=== fetching graphs from the 5090 box ==="
for M in $MODELS; do
  for P in fp16 int8 fp8; do
    [ -f "$W/${M}_${P}.onnx" ] && continue
    scp -q "${GPU_HOST:?set GPU_HOST=user@address}:coco_journal/exports/coco_pilot/${M}_${P}.onnx" "$W/" \
      || echo "  missing ${M}_${P}.onnx"
  done
done
ls "$W"/*.onnx | wc -l

echo "=== building strongly-typed engines on Ada (TRT $($PY -c 'import tensorrt;print(tensorrt.__version__)')) ==="
$PY - <<'PYEOF'
import os, tensorrt as trt
W = "/tmp/claude-1000/-data-nvme-paper/4fb5f984-9af2-48bc-ac1a-4fd20496f557/scratchpad/ada"
L = trt.Logger(trt.Logger.ERROR)
fails = []
for f in sorted(os.listdir(W)):
    if not f.endswith(".onnx"):
        continue
    plan = os.path.join(W, f.replace(".onnx", ".plan"))
    if os.path.exists(plan):
        continue
    b = trt.Builder(L)
    n = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    p = trt.OnnxParser(n, L)
    if not p.parse(open(os.path.join(W, f), "rb").read()):
        fails.append((f, "parse")); print("  PARSE FAIL", f, flush=True); continue
    c = b.create_builder_config()
    c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    e = b.build_serialized_network(n, c)
    if e is None:
        fails.append((f, "build")); print("  BUILD FAIL", f, flush=True); continue
    open(plan, "wb").write(bytes(memoryview(e)))
    print("  ok", f, flush=True)
# A build failure is itself a result: it means the format is unavailable on
# this generation, which is exactly what the paper needs to know.
print("BUILD_FAILURES:", fails)
PYEOF

echo "=== randomised-order latency, 3 repeats ==="
$PY - <<'PYEOF'
import json, os, random, subprocess
W = "/tmp/claude-1000/-data-nvme-paper/4fb5f984-9af2-48bc-ac1a-4fd20496f557/scratchpad/ada"
PY = "/data_nvme/paper/.venv_pilot/bin/python"
models = ["yolo11n","yolo11m","yolo11x","yolov8n","yolov8m","yolov8x","yolo26n","yolo26m","yolo26x"]
jobs = [(m,p) for m in models for p in ("fp16","int8","fp8")]
out = "metrics/ada/latency640_ada.jsonl"
open(out,"w").close()
rng = random.Random(0)
for rep in range(3):
    order = jobs[:]; rng.shuffle(order)
    for m,p in order:
        eng = os.path.join(W, f"{m}_{p}.plan")
        if not os.path.exists(eng):
            continue
        r = subprocess.run([PY,"tools/measure_trt_ada.py","--engine",eng,"--imgsz","640",
                            "--n-warmup","100","--n-iters","500"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("FAIL",m,p,r.stderr[-200:], flush=True); continue
        d = json.loads(r.stdout); d.update(model=m, precision=p, repeat=rep)
        open(out,"a").write(json.dumps(d)+"\n")
    print(f"repeat {rep} done", flush=True)
print("->", out)
PYEOF
echo "ADA_LATENCY_DONE"
