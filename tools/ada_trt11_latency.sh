#!/bin/bash
# The cross-generation comparison with the compiler held fixed.
#
# The first Ada pass used the TensorRT that happens to be installed here
# (10.16), which confounds hardware generation with compiler version: the
# Blackwell table was built with 11.1. TensorRT 11.1 turns out to be
# installable on this driver via the CUDA-12 wheel, so this pass rebuilds and
# re-times the same nine graphs under 11.1 on Ada. Blackwell/11.1 versus
# Ada/11.1 differs in one variable.
#
# Waits for the 10.16 pass to finish first: a concurrent engine build would
# contaminate the latency measurements it is taking.
set -uo pipefail
cd /data_nvme/paper
PY=/data_nvme/paper/.venv_trt11/bin/python
W=/tmp/claude-1000/-data-nvme-paper/4fb5f984-9af2-48bc-ac1a-4fd20496f557/scratchpad/ada
T=$W/trt11
mkdir -p "$T" metrics/ada

echo "[trt11] waiting for the TensorRT 10.16 pass to finish..."
while ! grep -q ADA_LATENCY_DONE "$W/../ada.log" 2>/dev/null; do sleep 30; done
echo "[trt11] 10.16 pass done at $(date '+%F %T'); GPU is ours"
# let clocks settle after the previous run
sleep 60

echo "=== building under TensorRT $($PY -c 'import tensorrt;print(tensorrt.__version__)') on Ada ==="
$PY - <<'PYEOF'
import os, tensorrt as trt, time
W = "/tmp/claude-1000/-data-nvme-paper/4fb5f984-9af2-48bc-ac1a-4fd20496f557/scratchpad/ada"
T = os.path.join(W, "trt11")
L = trt.Logger(trt.Logger.ERROR)
fails = []
for f in sorted(x for x in os.listdir(W) if x.endswith(".onnx")):
    plan = os.path.join(T, f.replace(".onnx", ".plan"))
    if os.path.exists(plan):
        continue
    b = trt.Builder(L)
    n = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    p = trt.OnnxParser(n, L)
    if not p.parse(open(os.path.join(W, f), "rb").read()):
        fails.append((f, "parse")); print("  PARSE FAIL", f, flush=True); continue
    c = b.create_builder_config()
    c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    t0 = time.time(); e = b.build_serialized_network(n, c)
    if e is None:
        fails.append((f, "build")); print("  BUILD FAIL", f, flush=True); continue
    open(plan, "wb").write(bytes(memoryview(e)))
    print(f"  ok {f}  {time.time()-t0:.0f}s", flush=True)
print("TRT11_BUILD_FAILURES:", fails)
PYEOF

echo "=== randomised-order latency under 11.1, 3 repeats ==="
$PY - <<'PYEOF'
import json, os, random, subprocess
T = "/tmp/claude-1000/-data-nvme-paper/4fb5f984-9af2-48bc-ac1a-4fd20496f557/scratchpad/ada/trt11"
PY = "/data_nvme/paper/.venv_trt11/bin/python"
models = ["yolo11n","yolo11m","yolo11x","yolov8n","yolov8m","yolov8x","yolo26n","yolo26m","yolo26x"]
jobs = [(m,p) for m in models for p in ("fp16","int8","fp8")]
out = "metrics/ada/latency640_ada_trt11.jsonl"
open(out,"w").close()
rng = random.Random(0)
for rep in range(3):
    order = jobs[:]; rng.shuffle(order)
    for m,p in order:
        eng = os.path.join(T, f"{m}_{p}.plan")
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
echo "ADA_TRT11_DONE"
