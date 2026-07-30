"""Operating-point analysis: EWD(rho), recall@fixed-conf, and threshold recalibration.

Covers reviewer D3 (C2 is degenerate at rho=0.5 -> report a curve instead) and
D4 (the recall collapse comes from score compression, so re-tuning the
threshold should claw it back -> turns a "problem" finding into a recommendation).

Everything derives from ONE greedy matching pass per (model, precision), done at
the lowest conf floor. That is exact, not an approximation: greedy matching goes
by descending score, so dropping low-score detections can never change what a
higher-score detection matched. Filtering that single matched set by score >= c
therefore reproduces exactly what a run at conf=c would have produced, for any c
above the floor.

Outputs metrics/operating_point.json plus printed tables:
  * ewd_rho          : s*(rho) per precision, rho in {0.5..0.9}
  * pr_curve         : recall/precision vs conf (overall and per height bin)
  * score_percentiles: evidence for the score-compression mechanism
  * recalibration    : conf* where the quantized model regains FP32's recall,
                       and what precision it pays for it

Usage:
    python tools/operating_point.py --models yolo26n yolo11s yolov8s yolo11m \
        --precisions fp32 fp16 int8_ptq fp8 --out metrics/operating_point.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from common import REPO_ROOT, bin_of, load_size_bins

RHOS = (0.5, 0.6, 0.7, 0.8, 0.9)
CONF_GRID = (0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
             0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
REF_CONF = 0.25   # the stock Ultralytics operating point


def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_once(gt_dataset, dt_list, iou_thr=0.5):
    """Class-aware greedy match, per image, descending score.

    Returns (scores, is_tp, gt_height) aligned arrays over every detection, plus
    GT counts. gt_height is the height of the GT a TP matched (nan for FPs), so
    per-size-bin recall can be read straight off the same arrays.
    """
    gt_by_img = defaultdict(list)
    for a in gt_dataset["annotations"]:
        gt_by_img[a["image_id"]].append(
            {"category_id": a["category_id"], "bbox": a["bbox"],
             "h": a["height_px"], "matched": False})
    dt_by_img = defaultdict(list)
    for d in dt_list:
        dt_by_img[d["image_id"]].append(d)

    scores, is_tp, gt_h = [], [], []
    for img_id, dts in dt_by_img.items():
        gts = gt_by_img.get(img_id, [])
        for d in sorted(dts, key=lambda x: -x["score"]):
            best_iou, best_gt = 0.0, None
            for g in gts:
                if g["matched"] or g["category_id"] != d["category_id"]:
                    continue
                iou = _iou_xywh(g["bbox"], d["bbox"])
                if iou > best_iou:
                    best_iou, best_gt = iou, g
            scores.append(d["score"])
            if best_gt is not None and best_iou >= iou_thr:
                best_gt["matched"] = True
                is_tp.append(True)
                gt_h.append(best_gt["h"])
            else:
                is_tp.append(False)
                gt_h.append(np.nan)

    n_gt = len(gt_dataset["annotations"])
    heights = np.array([a["height_px"] for a in gt_dataset["annotations"]])
    return (np.array(scores), np.array(is_tp), np.array(gt_h), n_gt, heights)


def pr_at_conf(scores, is_tp, conf):
    keep = scores >= conf
    tp = int((is_tp & keep).sum())
    n_det = int(keep.sum())
    return tp, n_det


def recall_by_bin(gt_h_matched, scores, is_tp, conf, heights, size_bins):
    """Recall restricted to each height bin, at a given conf.

    NB: this stratifies AFTER a global match (a detection is matched against all
    GTs, then bucketed by the matched GT's height). That is NOT identical to
    eval_stratified.py's size-threshold recall, which restricts the GT set
    BEFORE matching — a detection overlapping two same-class GTs of different
    sizes can be assigned differently under the two schemes. Used here only for
    the operating-point view; do not present the two recall definitions as
    interchangeable.
    """
    keep = scores >= conf
    matched_h = gt_h_matched[keep & is_tp]
    out = {}
    for b in size_bins:
        lo, hi = b["min"], float(b["max"])
        n_gt_bin = int(((heights >= lo) & (heights < hi)).sum())
        n_hit = int(((matched_h >= lo) & (matched_h < hi)).sum())
        out[b["name"]] = {"recall": (n_hit / n_gt_bin) if n_gt_bin else float("nan"),
                          "n_gt": n_gt_bin, "n_hit": n_hit}
    return out


def s_star_from_curve(curve, rho):
    """Smallest size threshold s where recall(size >= s) first reaches rho."""
    for pt in curve:
        if pt["n_gt"] > 0 and pt["recall"] is not None and pt["recall"] >= rho:
            return pt["s_px"]
    return None


def main(models, precisions, gt_path, metrics_dir, out_path):
    with open(gt_path, encoding="utf-8") as f:
        gt_dataset = json.load(f)
    size_bins = load_size_bins()["height_bins"]

    results = {}
    for model in models:
        results[model] = {}
        for prec in precisions:
            pred_p = os.path.join(metrics_dir, f"pred_{model}_{prec}.json")
            met_p = os.path.join(metrics_dir, f"{model}_{prec}.json")
            if not os.path.exists(pred_p):
                print(f"  skip {model}/{prec}: no predictions")
                continue
            with open(pred_p, encoding="utf-8") as f:
                dt = json.load(f)

            scores, is_tp, gt_h, n_gt, heights = match_once(gt_dataset, dt)

            pr = []
            for c in CONF_GRID:
                tp, n_det = pr_at_conf(scores, is_tp, c)
                pr.append({"conf": c,
                           "recall": tp / n_gt if n_gt else float("nan"),
                           "precision": tp / n_det if n_det else float("nan"),
                           "n_det": n_det})

            entry = {
                "pr_curve": pr,
                "score_percentiles": {
                    str(p): float(np.percentile(scores, p)) for p in (50, 75, 90, 95, 99)
                } if scores.size else {},
                "n_det_total": int(scores.size),
                "recall_by_bin_at_ref_conf": recall_by_bin(
                    gt_h, scores, is_tp, REF_CONF, heights, size_bins),
            }
            # EWD(rho) reuses the size-sweep curve eval_stratified already stored
            if os.path.exists(met_p):
                with open(met_p, encoding="utf-8") as f:
                    met = json.load(f)
                curve = met.get("ewd", {}).get("curve", [])
                entry["ewd_rho"] = {str(r): s_star_from_curve(curve, r) for r in RHOS}
            results[model][prec] = entry

    # ---- threshold recalibration (reviewer D4) ----
    for model, per_prec in results.items():
        if "fp32" not in per_prec:
            continue
        ref = next(p for p in per_prec["fp32"]["pr_curve"] if abs(p["conf"] - REF_CONF) < 1e-9)
        for prec, e in per_prec.items():
            if prec == "fp32":
                continue
            target_recall = ref["recall"]
            # HIGHEST conf on the grid that still recovers FP32's recall (i.e.
            # give back the least precision needed). This is a grid approximation
            # on CONF_GRID, not the exact continuous threshold.
            cands = [p for p in e["pr_curve"] if p["recall"] >= target_recall]
            best = max(cands, key=lambda p: p["conf"]) if cands else None
            at_ref = next(p for p in e["pr_curve"] if abs(p["conf"] - REF_CONF) < 1e-9)
            e["recalibration"] = {
                "fp32_ref": {"conf": REF_CONF, "recall": ref["recall"], "precision": ref["precision"]},
                "at_ref_conf": {"conf": REF_CONF, "recall": at_ref["recall"],
                                "precision": at_ref["precision"]},
                "recovered": ({"conf": best["conf"], "recall": best["recall"],
                               "precision": best["precision"]} if best else None),
                "recall_gap_at_ref": ref["recall"] - at_ref["recall"],
                "precision_cost_when_recovered": (
                    ref["precision"] - best["precision"] if best else None),
            }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---- printed summaries ----
    print("\n=== EWD s*(rho) [px] — None = never reaches rho at any size ===")
    hdr = f"{'model':10}{'prec':10}" + "".join(f"{'rho=' + str(r):>10}" for r in RHOS)
    print(hdr)
    for model, per_prec in results.items():
        for prec, e in per_prec.items():
            if "ewd_rho" not in e:
                continue
            row = f"{model:10}{prec:10}"
            for r in RHOS:
                v = e["ewd_rho"][str(r)]
                row += f"{('None' if v is None else str(v)):>10}"
            print(row)

    print(f"\n=== Operating point conf={REF_CONF}: recall / precision / p90 score ===")
    print(f"{'model':10}{'prec':10}{'recall':>9}{'prec.':>9}{'p90score':>10}{'Δrecall vs FP32':>18}")
    for model, per_prec in results.items():
        base = None
        for prec in ("fp32", "fp16", "int8_ptq", "fp8"):
            if prec not in per_prec:
                continue
            e = per_prec[prec]
            at = next(p for p in e["pr_curve"] if abs(p["conf"] - REF_CONF) < 1e-9)
            if prec == "fp32":
                base = at["recall"]
            d = "" if prec == "fp32" else f"{at['recall'] - base:+.3f}"
            p90 = e["score_percentiles"].get("90", float("nan"))
            print(f"{model:10}{prec:10}{at['recall']:9.3f}{at['precision']:9.3f}{p90:10.3f}{d:>18}")

    print("\n=== Recalibration: can lowering conf recover FP32 recall? ===")
    print(f"{'model':10}{'prec':10}{'conf*':>8}{'recall':>9}{'prec.':>9}"
          f"{'vs FP32 prec.':>15}")
    for model, per_prec in results.items():
        for prec, e in per_prec.items():
            rc = e.get("recalibration")
            if not rc:
                continue
            if rc["recovered"] is None:
                print(f"{model:10}{prec:10}{'—':>8}{'—':>9}{'—':>9}"
                      f"{'CANNOT recover':>15}")
            else:
                r = rc["recovered"]
                print(f"{model:10}{prec:10}{r['conf']:8.3f}{r['recall']:9.3f}"
                      f"{r['precision']:9.3f}{-rc['precision_cost_when_recovered']:+15.3f}")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8_ptq", "fp8"])
    ap.add_argument("--gt", default=os.path.join(REPO_ROOT, "metrics", "gt_test.json"))
    ap.add_argument("--metrics-dir", default=os.path.join(REPO_ROOT, "metrics"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "metrics", "operating_point.json"))
    args = ap.parse_args()
    main(args.models, args.precisions, args.gt, args.metrics_dir, args.out)
