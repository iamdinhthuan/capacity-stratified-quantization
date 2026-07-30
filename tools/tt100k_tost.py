"""TOST equivalence for the TT100K primary study (same rule as coco_tost.py).

DIFF here is dAP(S) - dAP(XL) on the sign-height strata. Size-neutrality is
claimed only when the 90% CI lies within [-Delta, +Delta]. Reuses the
validated bootstrap machinery of sur_bootstrap.py.

Run from the TT100K working copy (needs metrics/pred_{model}_{prec}.json and
metrics/gt_test.json):
    python tools/tt100k_tost.py --models yolo11n yolo11s yolov8s \
        --precision int8_ptq --n-boot 2000 --delta 0.005
"""
import argparse
import json
import os

import numpy as np

from common import REPO_ROOT
from sur_bootstrap import BIG, HEIGHT_BINS, accumulate_ap, build_eval, self_check


def main(models, precision, gt_path, metrics_dir, n_boot, seed, delta, out_path):
    with open(gt_path, encoding="utf-8") as f:
        gt_dataset = json.load(f)
    h_rngs = [(lo * lo, BIG if np.isinf(hi) else hi * hi - 1e-3) for _, lo, hi in HEIGHT_BINS]
    names = [b[0] for b in HEIGHT_BINS]
    iS, iXL = names.index("S"), names.index("XL")

    res, check = {}, True
    print(f"{'model':10} {'DIFF':>8} {'90% CI':>22} {'95% CI':>22}  TOST@d  1.5d  2d")
    for m in models:
        with open(os.path.join(metrics_dir, f"pred_{m}_fp32.json"), encoding="utf-8") as f:
            dt_ref = json.load(f)
        with open(os.path.join(metrics_dir, f"pred_{m}_{precision}.json"), encoding="utf-8") as f:
            dt_q = json.load(f)
        e_ref = build_eval(gt_dataset, dt_ref, h_rngs, "height2")
        e_q = build_eval(gt_dataset, dt_q, h_rngs, "height2")
        if check:
            d = self_check(e_ref)
            print(f"  self-check: max|dAP| = {d:.2e} {'OK' if d < 1e-9 else 'MISMATCH'}")
            if d >= 1e-9:
                raise SystemExit("accumulate mismatch")
            check = False

        n_img = len(e_ref._paramsEval.imgIds)
        ap_r, _ = accumulate_ap(e_ref, list(range(n_img)))
        ap_q, _ = accumulate_ap(e_q, list(range(n_img)))
        point = float((ap_r[iS] - ap_q[iS]) - (ap_r[iXL] - ap_q[iXL]))

        rng = np.random.default_rng(seed)
        draws = np.empty(n_boot)
        for b in range(n_boot):
            s = rng.choice(n_img, size=n_img, replace=True).tolist()
            f_r, _ = accumulate_ap(e_ref, s)
            f_q, _ = accumulate_ap(e_q, s)
            draws[b] = (f_r[iS] - f_q[iS]) - (f_r[iXL] - f_q[iXL])

        lo90, hi90 = np.percentile(draws, [5, 95])
        lo95, hi95 = np.percentile(draws, [2.5, 97.5])
        tost = {f"{k}delta": bool(np.percentile(draws, 5) > -k * delta and np.percentile(draws, 95) < k * delta)
                for k in (1.0, 1.5, 2.0)}
        res[m] = {"precision": precision, "n_boot": n_boot, "DIFF": point,
                  "ci90": [float(lo90), float(hi90)], "ci95": [float(lo95), float(hi95)],
                  "delta": delta, "tost_pass": tost, "draws": [float(x) for x in draws]}
        print(f"{m:10} {point:+8.4f} [{lo90:+.4f},{hi90:+.4f}] [{lo95:+.4f},{hi95:+.4f}]  "
              + "  ".join('PASS' if v else 'fail' for v in tost.values()))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precision", default="int8_ptq")
    ap.add_argument("--gt", default=os.path.join(REPO_ROOT, "metrics", "gt_test.json"))
    ap.add_argument("--metrics-dir", default=os.path.join(REPO_ROOT, "metrics"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.005)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "metrics", "tt100k_tost.json"))
    args = ap.parse_args()
    main(args.models, args.precision, args.gt, args.metrics_dir, args.n_boot, args.seed,
         args.delta, args.out)
