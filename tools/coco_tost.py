"""TOST equivalence for DIFF, as pre-registered (PREREGISTRATION §5).

Size-neutrality / size-fairness may be claimed ONLY when the 90% CI of
DIFF = dAP_small - dAP_large lies entirely within [-Delta, +Delta].
This script computes that 90% interval (plus the 95% one for reporting and
the raw bootstrap draws for any later interval), and prints the TOST verdict
at Delta, 1.5*Delta and 2*Delta.

Reuses the validated machinery of coco_boot_diff.py (which self-checks its
accumulate against stock pycocotools before use).

Usage:
    python tools/coco_tost.py --models yolo11n yolo11s yolo11m yolo11l yolo11x \
        --precision fp8 --n-boot 2000 --delta 0.005
"""
import argparse
import json
import os

import numpy as np
from pycocotools.coco import COCO

from coco_boot_diff import BINS, build_eval_native
from coco_common import GT_VAL, PILOT_METRICS
from sur_bootstrap import accumulate_ap, self_check


def draws_for(model, precision, reference, coco_gt, n_boot, seed, do_self_check):
    def load(prec):
        with open(os.path.join(PILOT_METRICS, f"pred_{model}_{prec}.json"), encoding="utf-8") as f:
            return json.load(f)

    e_ref = build_eval_native(coco_gt, load(reference), 100)
    e_q = build_eval_native(coco_gt, load(precision), 100)
    if do_self_check:
        d = self_check(e_ref)
        print(f"  self-check vs stock accumulate: max|dAP| = {d:.2e} {'OK' if d < 1e-9 else 'MISMATCH'}")
        if d >= 1e-9:
            raise SystemExit("custom accumulate does not match pycocotools")

    names = [n for n, _, _ in BINS]
    iS, iL = names.index("small"), names.index("large")
    n_img = len(e_ref._paramsEval.imgIds)
    full = list(range(n_img))
    ap_r, _ = accumulate_ap(e_ref, full)
    ap_q, _ = accumulate_ap(e_q, full)
    point = float((ap_r[iS] - ap_q[iS]) - (ap_r[iL] - ap_q[iL]))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(n_img, size=n_img, replace=True).tolist()
        f_r, _ = accumulate_ap(e_ref, s)
        f_q, _ = accumulate_ap(e_q, s)
        draws[b] = (f_r[iS] - f_q[iS]) - (f_r[iL] - f_q[iL])
    return point, draws


def verdict(draws, delta):
    lo, hi = np.percentile(draws, [5, 95])
    return bool(lo > -delta and hi < delta), float(lo), float(hi)


def main(models, precision, reference, n_boot, seed, delta, out_path):
    coco_gt = COCO(GT_VAL)
    res, check = {}, True
    print(f"{'model':10} {'DIFF':>8} {'90% CI':>22} {'95% CI':>22}  TOST@d  1.5d  2d")
    for m in models:
        point, draws = draws_for(m, precision, reference, coco_gt, n_boot, seed, check)
        check = False
        lo90, hi90 = np.percentile(draws, [5, 95])
        lo95, hi95 = np.percentile(draws, [2.5, 97.5])
        v1, _, _ = verdict(draws, delta)
        v15, _, _ = verdict(draws, 1.5 * delta)
        v2, _, _ = verdict(draws, 2 * delta)
        res[m] = {
            "precision": precision, "reference": reference, "n_boot": n_boot,
            "DIFF": point,
            "ci90": [float(lo90), float(hi90)], "ci95": [float(lo95), float(hi95)],
            "delta": delta,
            "tost_pass": {"1.0delta": v1, "1.5delta": v15, "2.0delta": v2},
            "draws": [float(x) for x in draws],
        }
        print(f"{m:10} {point:+8.4f} [{lo90:+.4f},{hi90:+.4f}] [{lo95:+.4f},{hi95:+.4f}]  "
              f"{'PASS' if v1 else 'fail':>5} {'PASS' if v15 else 'fail':>5} {'PASS' if v2 else 'fail':>5}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precision", default="fp8")
    ap.add_argument("--reference", default="fp32")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.005)
    ap.add_argument("--out", default=os.path.join(PILOT_METRICS, "tost.json"))
    args = ap.parse_args()
    main(args.models, args.precision, args.reference, args.n_boot, args.seed, args.delta, args.out)
