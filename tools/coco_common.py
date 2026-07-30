"""Shared helpers for the COCO pilot (README_journal.md v2 §3.4).

Everything COCO-specific lives here so the TT100K harness (common.py) stays
untouched: paths, the COCO80->COCO91 category-id mapping, Ultralytics-style
letterbox preprocessing, and a numpy per-class NMS that replicates the decode
of tools/infer_trt_raw.py (strict conf >, per-class NMS iou 0.7, global
top-300 cap, clip to frame) without needing torch in the pilot venv.
"""
import json
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_IMAGES_VAL = os.path.join(REPO_ROOT, "data", "coco", "images", "val2017")
COCO_CALIB_DIR = os.path.join(REPO_ROOT, "data", "coco", "calib", "images")
ANN_DIR = os.path.join(REPO_ROOT, "data", "coco_src", "annotations")
GT_VAL = os.path.join(ANN_DIR, "instances_val2017.json")
GT_TRAIN = os.path.join(ANN_DIR, "instances_train2017.json")
PILOT_EXPORTS = os.path.join(REPO_ROOT, "exports", "coco_pilot")
PILOT_METRICS = os.path.join(REPO_ROOT, "metrics", "coco_pilot")

# Ultralytics class index (0..79) -> official COCO category_id (1..90).
COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
                39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
                76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

MAX_DET = 300      # Ultralytics default cap (kept identical to infer_trt_raw.py)
CONF_FLOOR = 1e-3
NMS_IOU = 0.7


def val_images():
    """[(image_id, abs_path), ...] for every val2017 image, sorted by id."""
    with open(GT_VAL, encoding="utf-8") as f:
        gt = json.load(f)
    out = []
    for im in gt["images"]:
        p = os.path.join(COCO_IMAGES_VAL, im["file_name"])
        out.append((im["id"], p))
    out.sort()
    return out


def letterbox(im, new_shape=640, color=114):
    """Ultralytics-style letterbox to a SQUARE new_shape x new_shape input.

    Returns (padded_image, gain, (pad_w, pad_h)). Engines are static
    (1,3,640,640), so unlike Ultralytics' auto=True path for .pt models we
    always pad to the full square — the same choice for every precision rung,
    so it cancels in every FP32-vs-quantized comparison.
    """
    import cv2
    h, w = im.shape[:2]
    gain = min(new_shape / h, new_shape / w)
    nh, nw = round(h * gain), round(w * gain)
    if (nw, nh) != (w, h):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = (new_shape - nw) / 2, (new_shape - nh) / 2
    top, bottom = round(ph - 0.1), round(ph + 0.1)
    left, right = round(pw - 0.1), round(pw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                            value=(color, color, color))
    return im, gain, (left, top)


def preprocess(img_path, imgsz):
    """jpg path -> (1,3,imgsz,imgsz) float32 [0,1] RGB tensor + undo info."""
    import cv2
    im0 = cv2.imread(img_path)
    h0, w0 = im0.shape[:2]
    im, gain, (padx, pady) = letterbox(im0, imgsz)
    im = im[:, :, ::-1].transpose(2, 0, 1)
    return (np.ascontiguousarray(im, dtype=np.float32)[None] / 255.0,
            gain, padx, pady, w0, h0)


def _nms_numpy(boxes, scores, iou_thr):
    """Greedy NMS identical in semantics to torchvision.ops.nms."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1:][iou <= iou_thr]
    return keep


def decode_end2end(out, conf_thr, gain, padx, pady, ow, oh):
    """(1, max_det, 6) NMS-free e2e head (YOLO26): rows are
    [x1,y1,x2,y2,score,cls] in input pixels — threshold, undo letterbox, clip.
    Mirrors infer_trt_raw._decode_end2end."""
    rows = []
    for x1, y1, x2, y2, score, cls_id in out[0]:
        if score <= conf_thr:
            continue
        rx1 = (x1 - padx) / gain
        ry1 = (y1 - pady) / gain
        rx2 = (x2 - padx) / gain
        ry2 = (y2 - pady) / gain
        rx1, rx2 = max(rx1, 0.0), min(rx2, ow)
        ry1, ry2 = max(ry1, 0.0), min(ry2, oh)
        rows.append((float(rx1), float(ry1), float(rx2), float(ry2), float(score), int(cls_id)))
    rows.sort(key=lambda r: -r[4])
    return rows[:MAX_DET]


def decode_output(out, conf_thr, gain, padx, pady, ow, oh):
    """Dispatch on head layout: (1,N,6) e2e vs (1,4+nc,anchors) classic."""
    if out.shape[-1] == 6:
        return decode_end2end(out, conf_thr, gain, padx, pady, ow, oh)
    return decode_classic(out, conf_thr, gain, padx, pady, ow, oh)


def decode_classic(out, conf_thr, gain, padx, pady, ow, oh):
    """(1, 4+nc, anchors) raw head -> [(x1,y1,x2,y2,score,cls), ...] in original
    image pixels. Mirrors infer_trt_raw._decode_classic (argmax label, strict
    conf >, per-class NMS, global top-MAX_DET, clip), plus letterbox undo."""
    pred = out[0].T  # (anchors, 4+nc)
    cxcywh, cls_scores = pred[:, :4], pred[:, 4:]
    cls_id = cls_scores.argmax(axis=1)
    score = cls_scores.max(axis=1)
    keep = score > conf_thr
    if not keep.any():
        return []
    cxcywh, cls_id, score = cxcywh[keep], cls_id[keep], score[keep]

    boxes = np.empty((len(cxcywh), 4), dtype=np.float32)
    boxes[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
    boxes[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
    boxes[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
    boxes[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2

    results = []
    for c in np.unique(cls_id):
        m = cls_id == c
        for i in _nms_numpy(boxes[m], score[m], NMS_IOU):
            bx1, by1, bx2, by2 = boxes[m][i]
            sc = float(score[m][i])
            # undo letterbox, then clip to the original frame
            x1 = (bx1 - padx) / gain
            y1 = (by1 - pady) / gain
            x2 = (bx2 - padx) / gain
            y2 = (by2 - pady) / gain
            x1, x2 = max(x1, 0.0), min(x2, ow)
            y1, y2 = max(y1, 0.0), min(y2, oh)
            results.append((x1, y1, x2, y2, sc, int(c)))
    results.sort(key=lambda r: -r[4])
    return results[:MAX_DET]
