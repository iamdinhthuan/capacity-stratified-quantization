"""Run a D-FINE TensorRT engine over COCO val2017 -> COCO-format predictions.

D-FINE twin of tools/coco_infer_trt.py. Differences, all verified against the
D-FINE repo (exports/dfine_track/D-FINE):
  - TWO inputs: `images` float32 (1,3,640,640) and `orig_target_sizes` int64
    (1,2) in [w, h] order (src/data/dataset/coco_dataset.py:190 sets
    orig_size = [w, h]; postprocessor multiplies xyxy by repeat -> (w,h,w,h)).
  - Preprocessing replicates their val pipeline EXACTLY (no letterbox):
    PIL open + convert("RGB") (torchvision CocoDetection._load_image) ->
    PIL BILINEAR resize to 640x640 (v2.Resize on a PIL image) ->
    pil_to_tensor float32 / 255 (ConvertPILImage). No mean/std normalization.
  - Outputs (labels, boxes, scores): postprocessing is embedded in the ONNX —
    boxes are xyxy in ABSOLUTE original-image pixels, already top-300,
    no NMS needed. Labels are CONTIGUOUS 0..79; mapped to official COCO 91-id
    space via COCO80_TO_91 (verified equal to D-FINE's mscoco_label2category).
  - No box clipping (their eval does none), strict score > threshold.

Usage (venv_pilot):
    python dfine_infer_trt.py --engine engines/dfine_n_fp32.plan \
        --out /data_nvme/paper/metrics/coco_pilot/pred_dfine_n_fp32.json [--limit 200]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import tensorrt as trt
from PIL import Image

try:
    import cuda.bindings.runtime as cudart
except ImportError:
    import cuda.cudart as cudart

sys.path.insert(0, "/data_nvme/paper/tools")
from coco_common import COCO80_TO_91, val_images

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def cuda_check(err):
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def preprocess(img_path, imgsz):
    """Exact D-FINE val preprocessing. Returns (1,3,imgsz,imgsz) float32 + (w0,h0)."""
    im = Image.open(img_path).convert("RGB")
    w0, h0 = im.size
    im = im.resize((imgsz, imgsz), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(x), w0, h0


def run(engine_path, imgsz, conf, out_path, limit):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    device_buffers, host_buffers = {}, {}
    for name in io_names:
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host_buffers[name] = np.zeros(shape, dtype=dtype)
        err, ptr = cudart.cudaMalloc(host_buffers[name].nbytes)
        cuda_check(err)
        device_buffers[name] = ptr
        context.set_tensor_address(name, ptr)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, shape)
        print(f"  io {name}: {shape} {np.dtype(dtype).name} "
              f"({'input' if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else 'output'})")

    for req in ("images", "orig_target_sizes", "labels", "boxes", "scores"):
        assert req in io_names, f"engine lacks tensor {req}: {io_names}"

    err, stream = cudart.cudaStreamCreate()
    cuda_check(err)

    def h2d(name, arr):
        assert arr.dtype == host_buffers[name].dtype and arr.shape == host_buffers[name].shape, \
            f"{name}: feed {arr.shape}/{arr.dtype} vs engine {host_buffers[name].shape}/{host_buffers[name].dtype}"
        err, = cudart.cudaMemcpyAsync(device_buffers[name], arr.ctypes.data, arr.nbytes,
                                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cuda_check(err)

    def d2h(name):
        buf = host_buffers[name]
        err, = cudart.cudaMemcpyAsync(buf.ctypes.data, device_buffers[name], buf.nbytes,
                                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cuda_check(err)
        return buf

    size_dtype = host_buffers["orig_target_sizes"].dtype

    images = val_images()
    if limit:
        images = images[:limit]
    predictions = []
    t0 = time.time()
    for k, (img_id, img_path) in enumerate(images):
        inp, w0, h0 = preprocess(img_path, imgsz)
        sizes = np.ascontiguousarray(np.array([[w0, h0]], dtype=size_dtype))
        h2d("images", inp)
        h2d("orig_target_sizes", sizes)
        context.execute_async_v3(stream_handle=stream)
        labels, boxes, scores = d2h("labels"), d2h("boxes"), d2h("scores")
        cudart.cudaStreamSynchronize(stream)

        for lab, (x1, y1, x2, y2), sc in zip(labels[0], boxes[0], scores[0]):
            if sc <= conf:
                continue
            predictions.append({
                "image_id": int(img_id),
                "category_id": COCO80_TO_91[int(lab)],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(sc),
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
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only first N images (smoke test)")
    args = ap.parse_args()
    run(args.engine, args.imgsz, args.conf, args.out, args.limit)
