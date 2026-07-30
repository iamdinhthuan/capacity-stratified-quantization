"""Pilot headline statistic: DIFF = ΔAP_small − ΔAP_large on COCO, with a
paired image-level bootstrap CI (README_journal.md v2 §3.4 decision gate).

Reuses sur_bootstrap.py's validated trimmed accumulate (self-checked against
stock pycocotools at import of each model), with two protocol changes for
COCO: NATIVE area strata (GT keeps its mask 'area'; detections get bbox area
from loadRes — exactly what published AP_S/M/L means) and maxDets=100.

Usage:
    python tools/coco_boot_diff.py --models yolo11n yolo11s yolo11m yolo11l yolo11x \
        --precision int8 --reference fp32 --n-boot 500
"""
import argparse
import copy
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from coco_common import GT_VAL, PILOT_METRICS
from sur_bootstrap import accumulate_ap, self_check

BINS = [("all", 0, 1e10), ("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, 1e10)]


def build_eval_native(coco_gt, dt_list, max_dets):
    coco_dt = coco_gt.loadRes(copy.deepcopy(dt_list))
    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    e.params.areaRng = [[lo, hi] for _, lo, hi in BINS]
    e.params.areaRngLbl = [n for n, _, _ in BINS]
    e.params.maxDets = [max_dets]
    e.evaluate()
    return e


def run_model(model, precision, reference, coco_gt, n_boot, seed, do_self_check):
    def load(prec):
        p = os.path.join(PILOT_METRICS, f"pred_{model}_{prec}.json")
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    e_ref = build_eval_native(coco_gt, load(reference), 100)
    e_q = build_eval_native(coco_gt, load(precision), 100)

    if do_self_check:
        d = self_check(e_ref)
        print(f"  self-check vs stock accumulate: max|dAP| = {d:.2e} {'OK' if d < 1e-9 else 'MISMATCH'}")
        if d >= 1e-9:
            raise SystemExit("custom accumulate does not match pycocotools — aborting")

    n_img = len(e_ref._paramsEval.imgIds)
    full = list(range(n_img))
    ap_r, _ = accumulate_ap(e_ref, full)
    ap_q, _ = accumulate_ap(e_q, full)

    names = [n for n, _, _ in BINS]
    iS, iL = names.index("small"), names.index("large")

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        s = rng.choice(n_img, size=n_img, replace=True).tolist()
        f_r, _ = accumulate_ap(e_ref, s)
        f_q, _ = accumulate_ap(e_q, s)
        deltas.append(f_r - f_q)
    deltas = np.array(deltas)  # (n_boot, 4)

    def ci(v):
        v = v[np.isfinite(v)]
        return [float(x) for x in np.percentile(v, [2.5, 50, 97.5])] if v.size else [float("nan")] * 3

    diff_pt = float((ap_r[iS] - ap_q[iS]) - (ap_r[iL] - ap_q[iL]))
    diff_ci = ci(deltas[:, iS] - deltas[:, iL])
    result = {
        "model": model, "precision": precision, "reference": reference,
        "n_boot": n_boot, "n_images": n_img, "maxDets": 100,
        "ap_ref": {n: float(ap_r[i]) for i, n in enumerate(names)},
        "ap_q": {n: float(ap_q[i]) for i, n in enumerate(names)},
        "delta": {n: float(ap_r[i] - ap_q[i]) for i, n in enumerate(names)},
        "DIFF_small_minus_large": diff_pt,
        "ci_DIFF": diff_ci,
        "ci_delta": {n: ci(deltas[:, i]) for i, n in enumerate(names)},
    }
    verdict = "EXCLUDES 0" if (diff_ci[0] > 0 or diff_ci[2] < 0) else "includes 0"
    print(f"  {model}: dAP_S={result['delta']['small']:+.4f} dAP_L={result['delta']['large']:+.4f} "
          f"DIFF={diff_pt:+.4f} CI95=[{diff_ci[0]:+.4f},{diff_ci[2]:+.4f}] {verdict}")
    return result


def main(models, precision, reference, n_boot, seed, out_path):
    coco_gt = COCO(GT_VAL)
    all_res, check = {}, True
    for m in models:
        print(f"\n### {m} ({reference} vs {precision}), {n_boot} resamples")
        all_res[m] = run_model(m, precision, reference, coco_gt, n_boot, seed, check)
        check = False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--reference", default="fp32")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(PILOT_METRICS, "boot_diff.json"))
    args = ap.parse_args()
    main(args.models, args.precision, args.reference, args.n_boot, args.seed, args.out)
