"""Run a RAW TensorRT engine (built by tools/build_engine.py) over test images.

Two output layouts confirmed live (2026-07-15), auto-detected by shape:

- End-to-end heads (e.g. yolo26n): "output0" shape (1, 300, 6), already-NMS'd
  [x1, y1, x2, y2, score, class_id] rows in imgsz x imgsz INPUT pixel coords.
  Verified by running a real image through and checking value ranges by hand.

- Classic (non-end2end) heads (e.g. yolo11m): "output0" shape
  (1, 4+nc, num_anchors) — dense per-anchor [cx, cy, w, h, class_scores...],
  boxes in xywh INPUT pixel coords, class scores already sigmoid'd (range
  0-1, no extra activation needed). Needs manual xywh->xyxy + argmax over
  classes + per-class NMS (torchvision.ops.nms, iou=0.7 matching Ultralytics'
  own default) since anchors aren't pre-suppressed. Verified the same way.

decode_output() picks the path from the output tensor's last dim (== 6 ->
end2end, else -> classic), reading the first output tensor by value (the
torch-native modelopt export names it "output", the CLI one "output0"). Both
paths threshold with strict conf >, clip to the frame, and cap at 300
detections to match Ultralytics postprocessing, so a raw FP8 comparison
reflects the precision change rather than a decode difference. If a future head
uses yet another layout, this needs revisiting — it is not a generic decoder.

Preprocess uses a plain resize (no letterbox), which only preserves aspect
ratio because TT100K images are square (2048x2048); boxes are rescaled back
to each image's own original size via (orig_w/imgsz, orig_h/imgsz), read
per-image so this still works if a non-square image ever shows up.
"""
import argparse
import json
import os

import numpy as np
import tensorrt as trt
import torch
from torchvision.ops import nms

try:
    import cuda.bindings.runtime as cudart
except ImportError:
    import cuda.cudart as cudart

from common import REPO_ROOT, image_size, list_images

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def cuda_check(err):
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


MAX_DET = 300      # Ultralytics default cap on detections kept per image
CONF_FLOOR = 1e-3  # detections below this are never emitted (matches infer.py --conf floor)


def _clip(x1, y1, x2, y2, ow, oh):
    return (min(max(x1, 0.0), ow), min(max(y1, 0.0), oh),
            min(max(x2, 0.0), ow), min(max(y2, 0.0), oh))


def _decode_end2end(out, conf_thr, sx, sy, ow, oh):
    # engine already ran NMS; just threshold, rescale, clip to frame.
    rows = []
    for x1, y1, x2, y2, score, cls_id in out[0]:  # (300, 6)
        if score <= conf_thr:  # strict >, matching Ultralytics
            continue
        cx1, cy1, cx2, cy2 = _clip(x1 * sx, y1 * sy, x2 * sx, y2 * sy, ow, oh)
        rows.append((cx1, cy1, cx2, cy2, float(score), int(cls_id)))
    rows.sort(key=lambda r: -r[4])
    return rows[:MAX_DET]


def _decode_classic(out, conf_thr, sx, sy, ow, oh, iou_thr=0.7):
    """Match Ultralytics' non-end2end postprocess: strict conf >, per-class NMS,
    a global top-MAX_DET cap, and clipping to the image frame — so a FP8 raw
    comparison reflects the precision change, not a postprocessing difference."""
    pred = out[0].T  # (num_anchors, 4+nc)
    cxcywh, cls_scores = pred[:, :4], pred[:, 4:]
    cls_id = cls_scores.argmax(axis=1)
    score = cls_scores.max(axis=1)
    keep = score > conf_thr  # strict, matching Ultralytics
    if not keep.any():
        return []
    cxcywh, cls_id, score = cxcywh[keep], cls_id[keep], score[keep]

    x1 = cxcywh[:, 0] - cxcywh[:, 2] / 2
    y1 = cxcywh[:, 1] - cxcywh[:, 3] / 2
    x2 = cxcywh[:, 0] + cxcywh[:, 2] / 2
    y2 = cxcywh[:, 1] + cxcywh[:, 3] / 2
    boxes = torch.from_numpy(np.stack([x1, y1, x2, y2], axis=1).astype(np.float32))
    scores_t = torch.from_numpy(score.astype(np.float32))
    cls_t = torch.from_numpy(cls_id.astype(np.int64))

    results = []
    for c in cls_t.unique().tolist():
        mask = cls_t == c
        idx = nms(boxes[mask], scores_t[mask], iou_thr)
        b, s = boxes[mask][idx], scores_t[mask][idx]
        for (bx1, by1, bx2, by2), sc in zip(b.tolist(), s.tolist()):
            cx1, cy1, cx2, cy2 = _clip(bx1 * sx, by1 * sy, bx2 * sx, by2 * sy, ow, oh)
            results.append((cx1, cy1, cx2, cy2, float(sc), int(c)))
    results.sort(key=lambda r: -r[4])       # global top-MAX_DET across classes
    return results[:MAX_DET]


def decode_output(raw_outputs, conf_thr, sx, sy, ow, oh):
    # single output tensor; the engine's own name is not always "output0"
    # (the torch-native modelopt export calls it "output"), so take the first
    # output rather than hardcoding a key.
    out = next(iter(raw_outputs.values()))
    if out.shape[-1] == 6:  # (1, max_det, 6) end2end vs (1, 4+nc, num_anchors) classic
        return _decode_end2end(out, conf_thr, sx, sy, ow, oh)
    return _decode_classic(out, conf_thr, sx, sy, ow, oh)


def preprocess(img_path, imgsz):
    import cv2
    im = cv2.imread(img_path)
    im = cv2.resize(im, (imgsz, imgsz))
    im = im[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(im, dtype=np.float32) / 255.0


def run(engine_path, images_dir, imgsz, conf, out_path):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    device_buffers = {}
    host_buffers = {}
    for name in io_names:
        shape = engine.get_tensor_shape(name)
        shape = tuple(1 if d == -1 else d for d in shape)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host_buffers[name] = np.empty(shape, dtype=dtype)
        nbytes = host_buffers[name].nbytes
        err, ptr = cudart.cudaMalloc(nbytes)
        cuda_check(err)
        device_buffers[name] = ptr
        context.set_tensor_address(name, ptr)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, shape)

    input_name = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
    files = list_images(images_dir)
    predictions = []

    err, stream = cudart.cudaStreamCreate()
    cuda_check(err)

    for img_id, img_path in enumerate(files):
        orig_w, orig_h = image_size(img_path)
        sx, sy = orig_w / imgsz, orig_h / imgsz
        inp = preprocess(img_path, imgsz)[None]
        err, = cudart.cudaMemcpyAsync(
            device_buffers[input_name], inp.ctypes.data, inp.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream,
        )
        cuda_check(err)
        context.execute_async_v3(stream_handle=stream)
        raw = {}
        for name in io_names:
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                buf = host_buffers[name]
                err, = cudart.cudaMemcpyAsync(
                    buf.ctypes.data, device_buffers[name], buf.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream,
                )
                cuda_check(err)
                raw[name] = buf
        cudart.cudaStreamSynchronize(stream)

        for x1, y1, x2, y2, score, cls_id in decode_output(raw, conf, sx, sy, orig_w, orig_h):
            predictions.append({
                "image_id": img_id, "category_id": int(cls_id),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"{len(predictions)} detections -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "metrics", "pred_raw.json"))
    args = ap.parse_args()
    run(args.engine, args.images_dir, args.imgsz, args.conf, args.out)
