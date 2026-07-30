"""Run a raw TensorRT engine over COCO val2017 -> COCO-format predictions.

The COCO twin of tools/infer_trt_raw.py with three differences:
  - letterbox preprocessing (COCO images are not square; TT100K's plain
    resize would distort aspect ratio) with exact undo on the way out;
  - real COCO image_ids and COCO80->91 category ids, so the output json
    drops straight into pycocotools against instances_val2017.json;
  - numpy NMS (same semantics as torchvision.ops.nms) so the pilot venv
    needs no torch.

Usage:
    python tools/coco_infer_trt.py --engine exports/coco_pilot/yolo11n_int8.plan \
        --out metrics/coco_pilot/pred_yolo11n_int8.json [--limit 200]
"""
import argparse
import json
import os
import time

import numpy as np
import tensorrt as trt

try:
    import cuda.bindings.runtime as cudart
except ImportError:
    import cuda.cudart as cudart

from coco_common import COCO80_TO_91, CONF_FLOOR, decode_output, preprocess, val_images

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def cuda_check(err):
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def run(engine_path, imgsz, conf, out_path, limit):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    device_buffers, host_buffers = {}, {}
    for name in io_names:
        shape = tuple(1 if d == -1 else d for d in engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host_buffers[name] = np.empty(shape, dtype=dtype)
        err, ptr = cudart.cudaMalloc(host_buffers[name].nbytes)
        cuda_check(err)
        device_buffers[name] = ptr
        context.set_tensor_address(name, ptr)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, shape)

    input_name = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
    output_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    err, stream = cudart.cudaStreamCreate()
    cuda_check(err)

    images = val_images()
    if limit:
        images = images[:limit]
    predictions = []
    t0 = time.time()
    for k, (img_id, img_path) in enumerate(images):
        inp, gain, padx, pady, w0, h0 = preprocess(img_path, imgsz)
        err, = cudart.cudaMemcpyAsync(device_buffers[input_name], inp.ctypes.data, inp.nbytes,
                                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cuda_check(err)
        context.execute_async_v3(stream_handle=stream)
        buf = host_buffers[output_names[0]]
        err, = cudart.cudaMemcpyAsync(buf.ctypes.data, device_buffers[output_names[0]], buf.nbytes,
                                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cuda_check(err)
        cudart.cudaStreamSynchronize(stream)

        for x1, y1, x2, y2, score, cls in decode_output(buf, conf, gain, padx, pady, w0, h0):
            predictions.append({
                "image_id": int(img_id),
                "category_id": COCO80_TO_91[cls],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
        if (k + 1) % 500 == 0:
            print(f"{k + 1}/{len(images)} ({(k + 1) / (time.time() - t0):.1f} img/s)", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"{len(predictions)} detections -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=CONF_FLOOR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only first N images (smoke test)")
    args = ap.parse_args()
    run(args.engine, args.imgsz, args.conf, args.out, args.limit)
