"""Diff a D-FINE TRT engine against the torch-CPU golden npz on 3 fixed images.

Usage (venv_pilot):
    python compare_engine_vs_ref.py --engine engines/dfine_n_fp32.plan --ref ref_torch_n.npz
"""
import argparse
import os

import numpy as np
import tensorrt as trt

try:
    import cuda.bindings.runtime as cudart
except ImportError:
    import cuda.cudart as cudart

from dfine_infer_trt import cuda_check, preprocess

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
IMAGES = [
    "/data_nvme/paper/data/coco/images/val2017/000000000139.jpg",
    "/data_nvme/paper/data/coco/images/val2017/000000000285.jpg",
    "/data_nvme/paper/data/coco/images/val2017/000000000632.jpg",
]


def main(engine_path, ref_path, imgsz):
    ref = np.load(ref_path)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    host, dev = {}, {}
    for name in names:
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host[name] = np.zeros(shape, dtype=dtype)
        err, ptr = cudart.cudaMalloc(host[name].nbytes)
        cuda_check(err)
        dev[name] = ptr
        context.set_tensor_address(name, ptr)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, shape)
    err, stream = cudart.cudaStreamCreate()
    cuda_check(err)

    for p in IMAGES:
        x, w0, h0 = preprocess(p, imgsz)
        sizes = np.array([[w0, h0]], dtype=host["orig_target_sizes"].dtype)
        for nm, arr in (("images", x), ("orig_target_sizes", np.ascontiguousarray(sizes))):
            err, = cudart.cudaMemcpyAsync(dev[nm], arr.ctypes.data, arr.nbytes,
                                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
            cuda_check(err)
        context.execute_async_v3(stream_handle=stream)
        for nm in ("labels", "boxes", "scores"):
            err, = cudart.cudaMemcpyAsync(host[nm].ctypes.data, dev[nm], host[nm].nbytes,
                                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
            cuda_check(err)
        cudart.cudaStreamSynchronize(stream)

        key = os.path.basename(p).split(".")[0]
        rl, rb, rs = ref[f"{key}_labels"], ref[f"{key}_boxes"], ref[f"{key}_scores"]
        # compare on top-50 by ref order (tail of 300 is noise-ranked, ordering may swap)
        k = 50
        lab_match = (host["labels"][0][:k].astype(np.int64) == rl[0][:k]).mean()
        ds = np.abs(host["scores"][0][:k] - rs[0][:k]).max()
        db = np.abs(host["boxes"][0][:k] - rb[0][:k]).max()
        print(f"{key}: label match {lab_match*100:.0f}% | max|dscore| {ds:.5f} | max|dbox| {db:.3f}px"
              f" | trt top3 s={host['scores'][0][:3]} l={host['labels'][0][:3]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    main(args.engine, args.ref, args.imgsz)
