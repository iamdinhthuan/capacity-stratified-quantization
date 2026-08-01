#!/usr/bin/env python3
"""Decoder for Ultralytics RT-DETR TensorRT output.

The export emits (1, 300, 6): one row per object query, holding normalised
cxcywh followed by an explicit score and class id. That is neither branch in
coco_common - decode_end2end expects (1, N, 6) as xyxy in input pixels, and
decode_classic expects (1, 4+nc, anchors). Feeding RT-DETR to either silently
produces boxes in the wrong place, which downstream looks exactly like
quantization damage.

Normalisation is against one of two frames, and they are not distinguishable
from the numbers alone: the padded 640x640 square the network saw, or the
original image. `mode` selects; the FP32 gate in rtdetr_gate.py decides which
by running both and keeping whichever reproduces the framework's own AP.
"""
import numpy as np

MAX_DET = 300
N_COCO_CLASSES = 80


def decode_rtdetr(out, conf_thr, ow, oh, mode="stretch", gain=1.0, padx=0.0,
                  pady=0.0, imgsz=640):
    """-> [(x1, y1, x2, y2, score, cls), ...] in original-image pixels."""
    pred = out[0]
    if pred.shape[0] < pred.shape[1]:        # (channels, queries) -> (queries, channels)
        pred = pred.T

    if pred.shape[1] == 6:                   # explicit score and class columns
        boxes = pred[:, :4]
        score = pred[:, 4]
        cls_id = pred[:, 5].astype(np.int64)
    else:                                    # class-score vector
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        cls_id = scores.argmax(axis=1)
        score = scores.max(axis=1)

    keep = score > conf_thr
    if not keep.any():
        return []
    boxes, cls_id, score = boxes[keep], cls_id[keep], score[keep]

    bad = (cls_id < 0) | (cls_id >= N_COCO_CLASSES)
    if bad.any():
        raise ValueError(
            f"class ids out of range: {sorted(set(cls_id[bad].tolist()))[:5]}. "
            f"The output layout is not what this decoder assumes "
            f"(saw shape {pred.shape}).")

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if mode == "stretch":                    # normalised against the original image
        x1 = (cx - w / 2) * ow
        y1 = (cy - h / 2) * oh
        x2 = (cx + w / 2) * ow
        y2 = (cy + h / 2) * oh
    else:                                    # normalised against the padded square
        x1 = ((cx - w / 2) * imgsz - padx) / gain
        y1 = ((cy - h / 2) * imgsz - pady) / gain
        x2 = ((cx + w / 2) * imgsz - padx) / gain
        y2 = ((cy + h / 2) * imgsz - pady) / gain

    x1 = np.clip(x1, 0, ow); x2 = np.clip(x2, 0, ow)
    y1 = np.clip(y1, 0, oh); y2 = np.clip(y2, 0, oh)

    order = np.argsort(-score)[:MAX_DET]     # one-to-one head: no NMS
    return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
             float(score[i]), int(cls_id[i])) for i in order]
