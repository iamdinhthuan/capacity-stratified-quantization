"""Mechanism analysis pack (R5) on saved predictions — no GPU needed.

For each (model, reference=fp32, quant=int8) pair on COCO val2017:
  A. Score compression: p50/p90 detection score per size stratum (detection
     stratum by its bbox area), FP32 vs INT8, plus det counts at conf>=0.25.
  B. Per-class small-object damage: per-category AP on the small stratum for
     FP32 and INT8 (pycocotools precision array), top drivers of the loss.
  C. TIDE-lite fate analysis of small GT objects that FP32 recalls (IoU>=.5,
     class-aware, conf>=0.25) but INT8 does not:
       - loc_drift : INT8 has same-class det at IoU in [0.1, 0.5)
       - cls_flip  : INT8 has wrong-class det at IoU >= 0.5
       - vanished  : neither
     (mirrors the localization-vs-classification split of Reg-PTQ/TIDE).

Usage:
    python tools/coco_analysis_pack.py --models yolo11n yolo11s yolo11m yolo11l yolo11x
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from coco_common import GT_VAL, PILOT_METRICS

S32, S96 = 32 ** 2, 96 ** 2
STRATA = [("small", 0, S32), ("medium", S32, S96), ("large", S96, float("inf"))]


def stratum_of(area):
    for n, lo, hi in STRATA:
        if lo <= area < hi:
            return n
    return "large"


def load_preds(model, prec):
    with open(os.path.join(PILOT_METRICS, f"pred_{model}_{prec}.json"), encoding="utf-8") as f:
        return json.load(f)


def score_compression(preds):
    out = {}
    by_stratum = defaultdict(list)
    for d in preds:
        w, h = d["bbox"][2], d["bbox"][3]
        by_stratum[stratum_of(w * h)].append(d["score"])
    for s, vals in by_stratum.items():
        v = np.array(vals)
        out[s] = {"p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
                  "n_ge_025": int((v >= 0.25).sum())}
    return out


def per_class_small_ap(coco_gt, preds):
    coco_dt = coco_gt.loadRes([dict(d) for d in preds])
    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    e.params.areaRng = [[0, S32]]
    e.params.areaRngLbl = ["small"]
    e.params.maxDets = [100]
    e.evaluate(); e.accumulate()
    prec = e.eval["precision"]  # [T,R,K,1,1]
    out = {}
    for k, cid in enumerate(e.params.catIds):
        s = prec[:, :, k, 0, 0]
        valid = s[s > -1]
        out[int(cid)] = float(valid.mean()) if valid.size else float("nan")
    return out


def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    u = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / u if u > 0 else 0.0


def greedy_match(gts, dets, iou_thr):
    """class-aware greedy match by descending score; returns matched gt idx set."""
    matched = set()
    for d in sorted(dets, key=lambda x: -x["score"]):
        best, best_iou = None, iou_thr
        for i, g in enumerate(gts):
            if i in matched or g["category_id"] != d["category_id"]:
                continue
            iou = _iou(g["bbox"], d["bbox"])
            if iou >= best_iou:
                best, best_iou = i, iou
        if best is not None:
            matched.add(best)
    return matched


def fate_of_lost_small(coco_gt, preds_ref, preds_q, conf=0.25):
    gt_small = defaultdict(list)
    for a in coco_gt.dataset["annotations"]:
        if not a.get("iscrowd") and a["area"] < S32:
            gt_small[a["image_id"]].append(a)
    ref_by_img, q_by_img = defaultdict(list), defaultdict(list)
    for d in preds_ref:
        if d["score"] >= conf:
            ref_by_img[d["image_id"]].append(d)
    for d in preds_q:
        if d["score"] >= conf:
            q_by_img[d["image_id"]].append(d)

    fates = {"kept": 0, "loc_drift": 0, "cls_flip": 0, "vanished": 0, "ref_recalled": 0}
    for img_id, gts in gt_small.items():
        m_ref = greedy_match(gts, ref_by_img.get(img_id, []), 0.5)
        m_q = greedy_match(gts, q_by_img.get(img_id, []), 0.5)
        fates["ref_recalled"] += len(m_ref)
        for i in m_ref:
            if i in m_q:
                fates["kept"] += 1
                continue
            g = gts[i]
            qdets = q_by_img.get(img_id, [])
            same_cls_low = any(d["category_id"] == g["category_id"] and 0.1 <= _iou(g["bbox"], d["bbox"]) < 0.5 for d in qdets)
            wrong_cls_hi = any(d["category_id"] != g["category_id"] and _iou(g["bbox"], d["bbox"]) >= 0.5 for d in qdets)
            if same_cls_low:
                fates["loc_drift"] += 1
            elif wrong_cls_hi:
                fates["cls_flip"] += 1
            else:
                fates["vanished"] += 1
    return fates


def main(models, quant, out_path):
    coco_gt = COCO(GT_VAL)
    cats = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}
    result = {}
    for m in models:
        print(f"\n### {m} (fp32 vs {quant})")
        ref, q = load_preds(m, "fp32"), load_preds(m, quant)
        sc_ref, sc_q = score_compression(ref), score_compression(q)
        cls_ref, cls_q = per_class_small_ap(coco_gt, ref), per_class_small_ap(coco_gt, q)
        drops = {cid: cls_ref[cid] - cls_q[cid] for cid in cls_ref
                 if np.isfinite(cls_ref[cid]) and np.isfinite(cls_q[cid])}
        top = sorted(drops.items(), key=lambda kv: -kv[1])[:10]
        fates = fate_of_lost_small(coco_gt, ref, q)
        lost = fates["ref_recalled"] - fates["kept"]
        print(f"  score p90 small: {sc_ref['small']['p90']:.3f} -> {sc_q['small']['p90']:.3f} | "
              f"large: {sc_ref['large']['p90']:.3f} -> {sc_q['large']['p90']:.3f}")
        print(f"  small GT recalled by FP32@.25: {fates['ref_recalled']}, lost by {quant}: {lost} "
              f"(loc_drift {fates['loc_drift']}, cls_flip {fates['cls_flip']}, vanished {fates['vanished']})")
        print(f"  top small-AP drops: " + ", ".join(f"{cats[c]} {d:+.3f}" for c, d in top[:5]))
        result[m] = {
            "score_compression": {"fp32": sc_ref, quant: sc_q},
            "per_class_small_ap_drop_top10": [{"class": cats[c], "drop": d} for c, d in top],
            "small_fates": fates,
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--out", default=os.path.join(PILOT_METRICS, "analysis_pack.json"))
    args = ap.parse_args()
    main(args.models, args.quant, args.out)
