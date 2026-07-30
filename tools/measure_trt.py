"""Uniform raw-TensorRT GPU-compute latency for fp16/int8/fp8 engines/plans.
Times execute_async_v3 + stream sync (pure GPU inference, input resident on device),
so fp16/int8/fp8 are apples-to-apples. Samples board power during the timed loop."""
import argparse, json, time, threading, subprocess
import numpy as np
import tensorrt as trt
try:
    import cuda.bindings.runtime as cudart
except ImportError:
    import cuda.cudart as cudart

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def cuda_check(err):
    if isinstance(err, tuple): err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")

def sample_power(stop_event, samples):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                timeout=2).decode().strip()
            samples.append(float(out.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--n-warmup", type=int, default=50)
    ap.add_argument("--n-iters", type=int, default=500)
    args = ap.parse_args()

    with open(args.engine, "rb") as f:
        data = f.read()
    rt = trt.Runtime(TRT_LOGGER)
    engine = rt.deserialize_cuda_engine(data)
    if engine is None:
        # Ultralytics .engine files prepend a 4-byte little-endian metadata length
        # followed by a JSON metadata blob before the serialized TRT engine.
        meta_len = int.from_bytes(data[:4], "little", signed=True)
        if 0 < meta_len < len(data):
            engine = rt.deserialize_cuda_engine(data[4 + meta_len:])
    if engine is None:
        raise RuntimeError(f"could not deserialize TRT engine from {args.engine}")
    context = engine.create_execution_context()
    io = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    ptrs = {}
    for name in io:
        shape = engine.get_tensor_shape(name)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, shape)
            shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        err, p = cudart.cudaMalloc(nbytes); cuda_check(err)
        ptrs[name] = p
        context.set_tensor_address(name, p)
    err, stream = cudart.cudaStreamCreate(); cuda_check(err)

    def run_once():
        context.execute_async_v3(stream_handle=stream)
        cudart.cudaStreamSynchronize(stream)

    for _ in range(args.n_warmup): run_once()

    power = []; stop = threading.Event()
    th = threading.Thread(target=sample_power, args=(stop, power), daemon=True); th.start()
    lat = []
    for _ in range(args.n_iters):
        t0 = time.perf_counter(); run_once(); lat.append((time.perf_counter()-t0)*1000)
    stop.set(); th.join(timeout=2)
    lat = np.array(lat)
    import os
    res = {
        "engine": args.engine,
        "latency_mean_ms": float(lat.mean()),
        "latency_p50_ms": float(np.percentile(lat,50)),
        "latency_p99_ms": float(np.percentile(lat,99)),
        "throughput_img_s": float(1000.0/lat.mean()),
        "power_w": float(np.mean(power)) if power else None,
        "engine_size_mb": os.path.getsize(args.engine)/(1024*1024),
        "n_iters": args.n_iters,
    }
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
