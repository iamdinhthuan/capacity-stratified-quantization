#!/bin/bash
# Measurement phase only: the TensorRT 11.1 engines are already built; the
# first attempt died because the venv had the CUDA-13 cuda-python against a
# CUDA-12 driver, so every cudaMalloc returned cudaErrorInsufficientDriver.
set -uo pipefail
cd /data_nvme/paper
/data_nvme/paper/.venv_trt11/bin/python - <<'PYEOF'
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
            print("NO ENGINE", m, p, flush=True); continue
        r = subprocess.run([PY,"tools/measure_trt_ada.py","--engine",eng,"--imgsz","640",
                            "--n-warmup","100","--n-iters","500"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("FAIL",m,p,r.stderr[-300:], flush=True); continue
        d = json.loads(r.stdout); d.update(model=m, precision=p, repeat=rep)
        open(out,"a").write(json.dumps(d)+"\n")
    print(f"repeat {rep} done", flush=True)
print("->", out)
PYEOF
echo "ADA_TRT11_MEASURE_DONE"
