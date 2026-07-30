"""Latency / throughput / power measurement under a defensible protocol (reviewer B#3 / D5).

The previous numbers in this repo are not publishable and are discarded: they
were taken while another training job had the GPU, with no warm-up discipline,
and the power figure came from a loop that ran for well under a second — long
enough to catch the clock ramp, not steady state (that is why the FP8 pass once
reported 17.7 W against 87-161 W for the others: a methodology artifact, not a
real efficiency win).

What this does instead:
  * refuses to run unless the GPU is idle (no other compute processes) — checked
    again at the end, so a job that starts mid-run invalidates the measurement
    instead of silently corrupting it;
  * ONE measurement path for every precision. All engines, including the
    Ultralytics-exported .engine files (which carry a 4-byte length + JSON
    metadata header before the serialized engine) and the raw trtexec/API .plan
    files, are executed through the same bare TensorRT call. Timing a PyTorch
    .pt against a TensorRT engine would report the framework change, not the
    precision change;
  * warm-up, then per-iteration timing with an explicit stream sync;
  * power sampled over a SUSTAINED loop with the ramp-up window discarded;
  * records driver/CUDA/TRT/clocks, because a latency table without them is not
    reproducible.

Clock locking (`nvidia-smi -lgc`) needs root, which we do not have on this
shared box, and it would also disturb other users. Instead the SM clock is
sampled throughout and its spread is reported: if the clock is stable across the
run, the measurement stands; if it swings, the JSON says so and the number
should be treated accordingly.

Usage:
    python tools/measure_system.py --engine runs/.../best.engine \
        --model yolo26n --precision fp16 --metrics-json metrics/yolo26n_fp16.json
"""
import argparse
import json
import os
import subprocess
import threading
import time

import numpy as np
import tensorrt as trt

try:
    import cuda.bindings.runtime as cudart
except ImportError:
    import cuda.cudart as cudart

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def cuda_check(err):
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def gpu_busy_processes():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        timeout=10).decode().strip()
    return [l for l in out.splitlines() if l.strip()]


def env_info():
    q = ("driver_version,name,clocks.max.sm,clocks.max.mem,"
         "power.limit,memory.total")
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], timeout=10).decode().strip()
    keys = ["driver", "gpu", "max_sm_clock", "max_mem_clock", "power_limit", "memory_total"]
    return dict(zip(keys, [v.strip() for v in out.split(",")]), tensorrt=trt.__version__)


def load_engine_bytes(path):
    """Ultralytics .engine files prepend 4-byte LE length + JSON metadata."""
    with open(path, "rb") as f:
        head = f.read(4)
        n = int.from_bytes(head, byteorder="little")
        meta = None
        if 0 < n < 1_000_000:
            try:
                meta = json.loads(f.read(n).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                meta = None
        if meta is None:
            f.seek(0)
        return f.read(), meta


def sampler(stop, samples):
    """Poll power AND clock; a stable clock is what makes the number reportable."""
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw,clocks.sm,temperature.gpu",
                 "--format=csv,noheader,nounits"], timeout=2).decode().strip().splitlines()[0]
            p, c, t = [float(x) for x in out.split(",")]
            samples.append((time.perf_counter(), p, c, t))
        except Exception:
            pass
        time.sleep(0.1)


def measure(engine_path, imgsz, n_warmup, n_iters, power_seconds, ramp_discard):
    busy = gpu_busy_processes()
    if busy:
        raise SystemExit(f"GPU is not idle, refusing to measure. Running: {busy}")

    blob, meta = load_engine_bytes(engine_path)
    with trt.Runtime(TRT_LOGGER) as rt:
        engine = rt.deserialize_cuda_engine(blob)
    if engine is None:
        raise SystemExit(f"could not deserialize {engine_path}")
    ctx = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    dev, host = {}, {}
    for n in names:
        shape = tuple(1 if d == -1 else d for d in engine.get_tensor_shape(n))
        host[n] = np.zeros(shape, dtype=trt.nptype(engine.get_tensor_dtype(n)))
        err, ptr = cudart.cudaMalloc(host[n].nbytes)
        cuda_check(err)
        dev[n] = ptr
        ctx.set_tensor_address(n, ptr)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(n, shape)
    in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)

    err, stream = cudart.cudaStreamCreate()
    cuda_check(err)
    inp = np.zeros(host[in_name].shape, dtype=host[in_name].dtype)
    cudart.cudaMemcpyAsync(dev[in_name], inp.ctypes.data, inp.nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaStreamSynchronize(stream)

    for _ in range(n_warmup):
        ctx.execute_async_v3(stream_handle=stream)
    cudart.cudaStreamSynchronize(stream)

    lat = np.empty(n_iters)
    for i in range(n_iters):
        t0 = time.perf_counter()
        ctx.execute_async_v3(stream_handle=stream)
        cudart.cudaStreamSynchronize(stream)
        lat[i] = (time.perf_counter() - t0) * 1000.0

    # sustained loop for power: the GPU needs seconds to reach a steady clock
    samples = []
    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(stop, samples), daemon=True)
    th.start()
    t_start = time.perf_counter()
    n_sustained = 0
    while time.perf_counter() - t_start < power_seconds:
        ctx.execute_async_v3(stream_handle=stream)
        n_sustained += 1
        if n_sustained % 200 == 0:
            cudart.cudaStreamSynchronize(stream)
    cudart.cudaStreamSynchronize(stream)
    stop.set()
    th.join(timeout=2)

    steady = [(p, c, t) for (ts, p, c, t) in samples if ts - t_start >= ramp_discard]
    pw = np.array([s[0] for s in steady]) if steady else np.array([])
    ck = np.array([s[1] for s in steady]) if steady else np.array([])

    still_busy = gpu_busy_processes()
    contaminated = len(still_busy) > 1  # our own process shows up here

    res = {
        "engine": os.path.basename(engine_path),
        "engine_size_mb": os.path.getsize(engine_path) / 1e6,
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p90_ms": float(np.percentile(lat, 90)),
        "latency_p99_ms": float(np.percentile(lat, 99)),
        "latency_mean_ms": float(lat.mean()),
        "latency_std_ms": float(lat.std()),
        "fps_bs1": float(1000.0 / lat.mean()),
        "n_iters": n_iters,
        "n_warmup": n_warmup,
        "power_w_mean": float(pw.mean()) if pw.size else None,
        "power_w_std": float(pw.std()) if pw.size else None,
        "power_samples": int(pw.size),
        "power_window_s": power_seconds - ramp_discard,
        "sm_clock_mean": float(ck.mean()) if ck.size else None,
        "sm_clock_std": float(ck.std()) if ck.size else None,
        "sm_clock_stable": bool(ck.size and ck.std() < 0.02 * ck.mean()),
        "clock_locked": False,  # no root on this box; stability reported instead
        "gpu_exclusive": not contaminated,
        "env": env_info(),
        "trt_metadata": meta,
    }
    for n in names:
        cudart.cudaFree(dev[n])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help=".engine (Ultralytics) or .plan (raw)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--n-warmup", type=int, default=50)
    ap.add_argument("--n-iters", type=int, default=1000)
    ap.add_argument("--power-seconds", type=float, default=30.0)
    ap.add_argument("--ramp-discard", type=float, default=5.0)
    ap.add_argument("--metrics-json", help="merge result into this file's 'system' key")
    args = ap.parse_args()

    r = measure(args.engine, args.imgsz, args.n_warmup, args.n_iters,
                args.power_seconds, args.ramp_discard)
    print(json.dumps(r, indent=2))

    if not r["sm_clock_stable"]:
        print("WARNING: SM clock moved >2% during the run — treat latency/power as indicative")
    if args.metrics_json and os.path.exists(args.metrics_json):
        with open(args.metrics_json, encoding="utf-8") as f:
            d = json.load(f)
        d["system"] = r
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        print(f"merged into {args.metrics_json}")
