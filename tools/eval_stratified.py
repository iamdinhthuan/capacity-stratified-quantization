"""Stratified evaluation harness — the core of Giai đoạn 3 (README §8, §10).

Reads a GT COCO json (tools/gt_to_coco.py) and a predictions COCO json
(tools/infer.py) for ONE (detector, precision) engine and computes:

  1. overall mAP50 / mAP50-95
  2. mAP per height bin (XS..XL) and per COCO area bin (small/medium/large)
  3. mAP per super-class (prohibitory/warning/mandatory/other)
  4. mAP per class, restricted to the 45-class subset when reported
  5. Early-Warning Distance (EWD) + ΔEWD vs a baseline s* passed in

This script does NOT compare precisions against each other (no Δ/SUR) —
that cross-precision comparison happens in make_tables.py once
metrics/{model}_{precision}.json exists for every rung of the ladder.

Usage:
    python tools/eval_stratified.py --gt metrics/gt_test.json \
        --dt metrics/pred_yolo26n_fp32.json \
        --model yolo26n --precision fp32 \
        --out metrics/yolo26n_fp32.json
"""
import argparse
import copy
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import REPO_ROOT, load_classmap, load_size_bins


def _make_coco(dataset_dict):
    coco = COCO()
    coco.dataset = dataset_dict
    coco.createIndex()
    return coco


def _override_area(anns, mode):
    out = copy.deepcopy(anns)
    for a in out:
        x, y, w, h = a["bbox"]
        a["area"] = h * h if mode == "height2" else w * h
    return out


def _mean_precision(precision_arr, iou_lo=None, cat_idx=None):
    """precision_arr: [T,R,K,A,M] from COCOeval.eval['precision']."""
    s = precision_arr
    if iou_lo is not None:
        s = s[iou_lo:iou_lo + 1]
    if cat_idx is not None:
        s = s[:, :, cat_idx:cat_idx + 1]
    valid = s[s > -1]
    return float(valid.mean()) if valid.size else float("nan")


def _run_cocoeval(gt_dataset, dt_list, area_rng, area_mode="true", cat_ids=None):
    gt_ds = copy.deepcopy(gt_dataset)
    gt_ds["annotations"] = _override_area(gt_ds["annotations"], area_mode)
    coco_gt = _make_coco(gt_ds)

    dt_copy = _override_area(dt_list, area_mode) if dt_list else []
    coco_dt = coco_gt.loadRes(dt_copy) if dt_copy else coco_gt.loadRes([])

    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    e.params.areaRng = [list(area_rng)]
    e.params.areaRngLbl = ["all"]
    e.params.maxDets = [300]
    if cat_ids is not None:
        e.params.catIds = cat_ids
    e.evaluate()
    e.accumulate()
    return e, coco_gt


def overall_and_per_class(gt_dataset, dt_list, classes_45):
    e, coco_gt = _run_cocoeval(gt_dataset, dt_list, area_rng=(0, 1e10), area_mode="true")
    prec = e.eval["precision"]
    cat_ids = e.params.catIds  # order matches K axis
    overall = {
        "mAP50-95": _mean_precision(prec),
        "mAP50": _mean_precision(prec, iou_lo=0),
    }
    per_class = {}
    for k, cid in enumerate(cat_ids):
        per_class[cid] = {
            "mAP50-95": _mean_precision(prec, cat_idx=k),
            "mAP50": _mean_precision(prec, iou_lo=0, cat_idx=k),
        }
    per_class_45 = {cid: v for cid, v in per_class.items() if cid in classes_45}
    return overall, per_class_45


def per_height_bin(gt_dataset, dt_list, height_bins):
    out = {}
    for b in height_bins:
        hi = 1e10 if b["max"] in (float("inf"), ".inf") or b["max"] == float("inf") else b["max"]
        lo = b["min"]
        # Paper bins are half-open [lo, hi) but COCOeval's area filter is
        # inclusive on BOTH ends, so an instance with height exactly == hi
        # (area == hi^2) is counted in this bin AND the next. TT100K has 17 GTs
        # at exactly 96 px straddling L/XL. Shrink the upper edge by a small
        # epsilon so only the exact-boundary instance is pushed to the higher
        # bin. Heights are FRACTIONAL (8053/8181 non-integer), and the nearest
        # real area below a boundary sits only ~0.018 above it (h=11.999 -> area
        # 143.98 below the 144 edge), so the epsilon must be well under that gap:
        # 1e-3 excludes exact-integer boundaries yet keeps every fractional
        # neighbour. (Verified: 1e-3 reproduces the exact half-open counts
        # 16/1645/3452/2282/786; the earlier 0.5 wrongly dropped 5 XS + 3 S + 9 M.)
        hi_area = 1e10 if hi >= 1e10 else hi * hi - 1e-3
        rng = (lo * lo, hi_area)
        e, _ = _run_cocoeval(gt_dataset, dt_list, area_rng=rng, area_mode="height2")
        prec = e.eval["precision"]
        out[b["name"]] = {"mAP50-95": _mean_precision(prec), "mAP50": _mean_precision(prec, iou_lo=0)}
    return out


def per_coco_bin(gt_dataset, dt_list, coco_bins):
    # COCO small/medium/large are kept with pycocotools' native inclusive-both
    # boundaries on purpose: that IS the metric every paper reports, so the
    # boundary convention must match theirs, not our half-open height bins.
    out = {}
    for b in coco_bins:
        hi = 1e10 if b["max"] == float("inf") else b["max"]
        rng = (b["min"], hi)
        e, _ = _run_cocoeval(gt_dataset, dt_list, area_rng=rng, area_mode="true")
        prec = e.eval["precision"]
        out[b["name"]] = {"mAP50-95": _mean_precision(prec), "mAP50": _mean_precision(prec, iou_lo=0)}
    return out


def per_superclass(gt_dataset, dt_list, classmap):
    sc_names = sorted(set(classmap["superclass_prefix_map"].values()) | {classmap["default_superclass"]})
    sc_to_id = {name: i for i, name in enumerate(sc_names)}
    cat_id_to_sc = {c["id"]: sc_to_id[c["superclass"]] for c in gt_dataset["categories"]}

    gt_ds = copy.deepcopy(gt_dataset)
    gt_ds["categories"] = [{"id": i, "name": name} for name, i in sc_to_id.items()]
    for a in gt_ds["annotations"]:
        a["category_id"] = cat_id_to_sc[a["category_id"]]

    dt_remap = []
    for d in dt_list:
        if d["category_id"] in cat_id_to_sc:
            d2 = dict(d)
            d2["category_id"] = cat_id_to_sc[d["category_id"]]
            dt_remap.append(d2)

    e, _ = _run_cocoeval(gt_ds, dt_remap, area_rng=(0, 1e10), area_mode="true")
    prec = e.eval["precision"]
    cat_ids = e.params.catIds
    id_to_name = {i: name for name, i in sc_to_id.items()}
    out = {}
    for k, cid in enumerate(cat_ids):
        out[id_to_name[cid]] = {
            "mAP50-95": _mean_precision(prec, cat_idx=k),
            "mAP50": _mean_precision(prec, iou_lo=0, cat_idx=k),
        }
    return out


def _iou_xywh(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def recall_at_size_threshold(gt_dataset, dt_list, size_thr_px, conf_thr, iou_thr):
    """Class-aware recall over GT instances with height_px >= size_thr_px,
    using detections with score >= conf_thr, matched greedily by descending
    score at the given IoU threshold (single operating point, README §10)."""
    gt_by_img = {}
    n_gt = 0
    for a in gt_dataset["annotations"]:
        if a["height_px"] < size_thr_px:
            continue
        gt_by_img.setdefault(a["image_id"], []).append({"category_id": a["category_id"], "bbox": a["bbox"], "matched": False})
        n_gt += 1
    if n_gt == 0:
        return float("nan"), 0

    dt_by_img = {}
    for d in dt_list:
        if d["score"] < conf_thr:
            continue
        dt_by_img.setdefault(d["image_id"], []).append(d)

    n_matched = 0
    for img_id, gts in gt_by_img.items():
        dts = sorted(dt_by_img.get(img_id, []), key=lambda x: -x["score"])
        for d in dts:
            best_iou, best_gt = 0.0, None
            for g in gts:
                if g["matched"] or g["category_id"] != d["category_id"]:
                    continue
                iou = _iou_xywh(g["bbox"], d["bbox"])
                if iou > best_iou:
                    best_iou, best_gt = iou, g
            if best_gt is not None and best_iou >= iou_thr:
                best_gt["matched"] = True
                n_matched += 1
    return n_matched / n_gt, n_gt


def compute_ewd(gt_dataset, dt_list, ewd_cfg):
    rho = ewd_cfg["recall_threshold"]
    conf_thr = ewd_cfg["conf_threshold"]
    iou_thr = ewd_cfg["iou_threshold"]
    step = ewd_cfg["scan_step_px"]
    s_max = ewd_cfg["scan_max_px"]

    curve = []
    s_star = None
    for s in range(0, s_max + 1, step):
        recall, n_gt = recall_at_size_threshold(gt_dataset, dt_list, s, conf_thr, iou_thr)
        curve.append({"s_px": s, "recall": recall, "n_gt": n_gt})
        if s_star is None and n_gt > 0 and recall >= rho:
            s_star = s
    return {"s_star_px": s_star, "rho": rho, "conf_thr": conf_thr, "iou_thr": iou_thr, "curve": curve}


def evaluate(gt_path, dt_path, model_name, precision, out_path):
    with open(gt_path, encoding="utf-8") as f:
        gt_dataset = json.load(f)
    with open(dt_path, encoding="utf-8") as f:
        dt_list = json.load(f)

    classmap = load_classmap()
    size_bins = load_size_bins()
    classes_45 = {c["id"] for c in classmap["classes_45"]}

    overall, per_class_45 = overall_and_per_class(gt_dataset, dt_list, classes_45)
    height_bin_ap = per_height_bin(gt_dataset, dt_list, size_bins["height_bins"])
    coco_bin_ap = per_coco_bin(gt_dataset, dt_list, size_bins["coco_area_bins"])
    superclass_ap = per_superclass(gt_dataset, dt_list, classmap)
    ewd = compute_ewd(gt_dataset, dt_list, size_bins["ewd"])

    # Preserve any system metrics measure_system.py already merged: re-running
    # accuracy eval must NOT wipe a benchmark that took a locked-down idle-GPU
    # slot to produce. Only seed the null placeholder when the file is new.
    system = {"latency_p50_ms": None, "latency_p99_ms": None, "fps_bs1": None,
              "engine_size_mb": None, "power_w_mean": None}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                prev = json.load(f).get("system")
            if prev and any(v is not None for v in prev.values()):
                system = prev
        except (json.JSONDecodeError, OSError):
            pass

    result = {
        "model": model_name,
        "precision": precision,
        "overall": overall,
        "height_bin_ap": height_bin_ap,
        "coco_bin_ap": coco_bin_ap,
        "superclass_ap": superclass_ap,
        "per_class_45_ap": {classmap["names"][cid]: v for cid, v in per_class_45.items()},
        "ewd": ewd,
        "system": system,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"model": model_name, "precision": precision, "overall": overall,
                       "height_bin_mAP50": {k: v["mAP50"] for k, v in height_bin_ap.items()},
                       "ewd_s_star_px": ewd["s_star_px"]}, indent=2))
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "metrics", "result.json"))
    args = ap.parse_args()
    evaluate(args.gt, args.dt, args.model, args.precision, args.out)
