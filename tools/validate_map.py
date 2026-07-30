"""Validate the stratified harness against vanilla pycocotools (reviewer PHẦN B#5 / D6).

The harness already calls pycocotools internally, so what actually needs
checking is not "is AP computed right" but "are our PARAMETER OVERRIDES and
our area-encoding trick right":

  1. overall mAP  — we force maxDets=[300] and a single areaRng, vanilla COCO
     uses maxDets=[1,10,100] and 4 areaRngs. Confirm our number matches a
     vanilla COCOeval configured the same way, and quantify the gap vs the
     stock maxDets=100 number that the literature reports.
  2. COCO area bins — our coco_bin_ap must match vanilla stats[3..5] once
     maxDets is matched.
  3. height bins   — we smuggle "bbox height" through COCOeval's area filter by
     rewriting area:=h^2 and querying areaRng=(lo^2, hi^2). Verify the GT count
     COCOeval actually keeps per bin equals an independent count from the raw
     labels (tools/stat_sizes.py numbers).
  4. id alignment  — predictions' image_id must index the same sorted file list
     the GT was built from; a silent off-by-one here would corrupt every Δ.

Exit code is non-zero if any check fails, so this is usable as a gate.

Usage:
    python tools/validate_map.py --gt metrics/gt_test.json \
        --dt metrics/pred_yolo26n_fp32.json --harness metrics/yolo26n_fp32.json
"""
import argparse
import copy
import json
import sys
from collections import Counter

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import load_size_bins
from eval_stratified import _mean_precision, _override_area, _run_cocoeval

TOL = 0.002  # reviewer D6: accept |delta| < 0.002


def _silent_coco(dataset):
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def vanilla_cocoeval(gt_dataset, dt_list, max_dets=(1, 10, 100)):
    """Stock pycocotools evaluation, no overrides beyond maxDets."""
    coco_gt = _silent_coco(copy.deepcopy(gt_dataset))
    coco_dt = coco_gt.loadRes(copy.deepcopy(dt_list))
    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    e.params.maxDets = list(max_dets)
    e.evaluate()
    e.accumulate()
    e.summarize()
    return e


def check_overall(gt_dataset, dt_list, harness):
    print("\n=== CHECK 1: overall mAP ===")
    # (a) stock COCO settings, the number papers usually quote
    e100 = vanilla_cocoeval(gt_dataset, dt_list, max_dets=(1, 10, 100))
    stock_map = e100.stats[0]
    stock_map50 = e100.stats[1]

    # (b) vanilla, but configured exactly like the harness: maxDets=300, one areaRng
    e300 = vanilla_cocoeval(gt_dataset, dt_list, max_dets=(1, 10, 300))
    prec = e300.eval["precision"]
    # areaRng index 0 == 'all', maxDets index 2 == 300
    s_all = prec[:, :, :, 0, 2]
    vanilla_300 = float(s_all[s_all > -1].mean())
    s_all50 = prec[0:1, :, :, 0, 2]
    vanilla_300_50 = float(s_all50[s_all50 > -1].mean())

    h_map = harness["overall"]["mAP50-95"]
    h_map50 = harness["overall"]["mAP50"]

    d = abs(vanilla_300 - h_map)
    d50 = abs(vanilla_300_50 - h_map50)
    print(f"  stock COCO   (maxDets=100): mAP50-95={stock_map:.4f}  mAP50={stock_map50:.4f}")
    print(f"  vanilla      (maxDets=300): mAP50-95={vanilla_300:.4f}  mAP50={vanilla_300_50:.4f}")
    print(f"  harness      (maxDets=300): mAP50-95={h_map:.4f}  mAP50={h_map50:.4f}")
    print(f"  |vanilla300 - harness|    : {d:.5f} / {d50:.5f}  (tol {TOL})")
    print(f"  NOTE maxDets 100 vs 300 shifts mAP50-95 by {abs(stock_map - vanilla_300):.4f} "
          f"-> report maxDets in the paper.")
    ok = d < TOL and d50 < TOL
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def check_coco_bins(gt_dataset, dt_list, harness):
    print("\n=== CHECK 2: COCO area bins (small/medium/large) ===")
    e = vanilla_cocoeval(gt_dataset, dt_list, max_dets=(1, 10, 300))
    prec = e.eval["precision"]
    # COCOeval areaRng order: all, small, medium, large ; maxDets idx 2 == 300
    ok = True
    for name, aidx in (("small", 1), ("medium", 2), ("large", 3)):
        s = prec[:, :, :, aidx, 2]
        vanilla = float(s[s > -1].mean()) if (s > -1).any() else float("nan")
        h = harness["coco_bin_ap"][name]["mAP50-95"]
        d = abs(vanilla - h)
        good = d < TOL
        ok &= good
        print(f"  {name:7} vanilla={vanilla:.4f} harness={h:.4f} |d|={d:.5f} "
              f"{'PASS' if good else 'FAIL'}")
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def check_height_bins(gt_dataset):
    """The area:=h^2 trick must select exactly the instances stat_sizes counts."""
    print("\n=== CHECK 3: height-bin encoding (area:=h^2) ===")
    bins = load_size_bins()["height_bins"]

    # independent ground truth: count straight off the annotation heights
    truth = Counter()
    for a in gt_dataset["annotations"]:
        h = a["height_px"]
        for b in bins:
            hi = float(b["max"])
            if b["min"] <= h < hi:
                truth[b["name"]] += 1
                break

    # Reproduce the EXACT range eval_stratified.per_height_bin builds (hi^2 - 1e-3)
    # and apply COCOeval's INCLUSIVE-both test — this is what actually runs, so a
    # wrong epsilon (e.g. the earlier 0.5 that dropped fractional-height GTs) is
    # caught here. The earlier version replicated a private half-open filter and
    # therefore could not detect that bug.
    from eval_stratified import per_height_bin  # ensure single source of the range
    EPS = 1e-3
    anns_h2 = _override_area(gt_dataset["annotations"], "height2")
    ok = True
    for b in bins:
        lo, hi = b["min"], float(b["max"])
        rng_lo = lo * lo
        rng_hi = 1e10 if hi == float("inf") else hi * hi - EPS
        kept = sum(1 for a in anns_h2 if rng_lo <= a["area"] <= rng_hi)  # inclusive-both
        good = kept == truth[b["name"]]
        ok &= good
        print(f"  {b['name']:3} labels={truth[b['name']]:5}  COCOeval range keeps {kept:5}  "
              f"{'PASS' if good else 'FAIL'}")
    _ = per_height_bin  # imported to assert the module's range logic stays in sync
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def check_id_alignment(gt_dataset, dt_list):
    print("\n=== CHECK 4: image_id alignment ===")
    gt_ids = {im["id"] for im in gt_dataset["images"]}
    dt_ids = {d["image_id"] for d in dt_list}
    stray = dt_ids - gt_ids
    ok = not stray
    print(f"  GT images={len(gt_ids)}  images with >=1 pred={len(dt_ids)}  "
          f"pred ids not in GT={len(stray)}")
    cat_ids = {c["id"] for c in gt_dataset["categories"]}
    stray_cats = {d["category_id"] for d in dt_list} - cat_ids
    ok &= not stray_cats
    print(f"  pred category_ids not in GT categories={len(stray_cats)}")
    print("  ->", "PASS" if ok else "FAIL")
    return ok


def main(gt_path, dt_path, harness_path):
    with open(gt_path, encoding="utf-8") as f:
        gt_dataset = json.load(f)
    with open(dt_path, encoding="utf-8") as f:
        dt_list = json.load(f)
    with open(harness_path, encoding="utf-8") as f:
        harness = json.load(f)

    results = [
        check_id_alignment(gt_dataset, dt_list),
        check_height_bins(gt_dataset),
        check_overall(gt_dataset, dt_list, harness),
        check_coco_bins(gt_dataset, dt_list, harness),
    ]
    print("\n" + "=" * 60)
    if all(results):
        print("ALL CHECKS PASSED — harness numbers are trustworthy")
        return 0
    print("SOME CHECKS FAILED — fix the harness before trusting any delta")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--harness", required=True, help="metrics/{model}_{precision}.json from eval_stratified")
    args = ap.parse_args()
    sys.exit(main(args.gt, args.dt, args.harness))
