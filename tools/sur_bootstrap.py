"""Bootstrap confidence intervals for the size-unfairness ratio (reviewer B#1 / D1).

Why this exists: SUR = Δ_XS / Δ_XL, and the XS bin holds only 16 test
instances. The A2 calib ablation already showed the symptom — the same model
(yolo26n) gave SUR=0.39 with one calib set and -0.04 with another. A negative
ratio means Δ_XS and Δ_XL disagree in sign, which is what small-sample noise
looks like, not a stable effect. So no SUR number is quotable without a CI.

Method: paired image-level bootstrap. We resample test IMAGES (not instances,
so within-image correlation is preserved) and recompute AP per size bin for
FP32 and for the quantized run under the SAME resample, so the Δ is paired.

Speed: pycocotools' evaluate() is the expensive part and does not depend on the
resample, so it runs once per (precision, area-encoding); each bootstrap
iteration only re-runs an accumulate restricted to the resampled image indices.
The accumulate here is a trimmed, vectorized re-implementation of
COCOeval.accumulate — verified against the stock one by --self-check.

Also reports, alongside the ratio:
  * per-bin Δ with CI (interpretable on its own, no division)
  * Δ_small − Δ_large (a DIFFERENCE, which unlike the ratio stays finite when
    the denominator approaches zero — recommended as the headline statistic)

Usage:
    python tools/sur_bootstrap.py --models yolo26n yolo11s yolov8s yolo11m \
        --precision int8_ptq --n-boot 1000 --out metrics/sur_bootstrap.json
"""
import argparse
import copy
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import REPO_ROOT, load_size_bins

# (name, lo, hi) in the units the matching area-encoding produces
HEIGHT_BINS = [("XS", 0, 12), ("S", 12, 24), ("M", 24, 48), ("L", 48, 96), ("XL", 96, np.inf)]
COCO_BINS = [("small", 0, 1024), ("medium", 1024, 9216), ("large", 9216, np.inf)]

BIG = 1e10


def _silent_coco(dataset):
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def _override_area(anns, mode):
    out = copy.deepcopy(anns)
    for a in out:
        x, y, w, h = a["bbox"]
        a["area"] = h * h if mode == "height2" else w * h
    return out


def build_eval(gt_dataset, dt_list, area_rngs, area_mode):
    """One evaluate() pass carrying every area range we will later query."""
    gt = copy.deepcopy(gt_dataset)
    gt["annotations"] = _override_area(gt["annotations"], area_mode)
    coco_gt = _silent_coco(gt)
    coco_dt = coco_gt.loadRes(_override_area(dt_list, area_mode))

    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    e.params.areaRng = [list(r) for r in area_rngs]
    e.params.areaRngLbl = [f"a{i}" for i in range(len(area_rngs))]
    e.params.maxDets = [300]
    e.evaluate()
    return e


def accumulate_ap(e, i_list):
    """mAP50-95 and mAP50 per areaRng, over the given positional image indices.

    Mirrors COCOeval.accumulate but (a) takes explicit image positions so the
    same image can appear more than once (bootstrap), and (b) replaces the
    Python precision-envelope loop with a vectorized reverse-cummax.
    """
    p = e.params
    T, R = len(p.iouThrs), len(p.recThrs)
    K, A = len(p.catIds), len(p.areaRng)
    maxDet = p.maxDets[0]
    I0, A0 = len(e._paramsEval.imgIds), len(e._paramsEval.areaRng)

    precision = -np.ones((T, R, K, A))
    for k in range(K):
        Nk = k * A0 * I0
        for a in range(A):
            Na = a * I0
            pairs = [(i, e.evalImgs[Nk + Na + i]) for i in i_list]
            pairs = [(i, x) for i, x in pairs if x is not None]
            if not pairs:
                continue
            dtScores = np.concatenate([x["dtScores"][0:maxDet] for _, x in pairs])
            # Tie-break deterministically by originating image index, not by the
            # (random) resample order. Stock COCOeval uses a stable sort, whose
            # tie order equals concatenation = image order; on the full ordered
            # set this secondary key is already monotonic so we still match it
            # exactly. On a bootstrap resample the same image multiset can arrive
            # in different orders, and without a canonical key that alone would
            # perturb AP whenever scores tie (common after quantization). lexsort
            # keys: primary -score (last arg), secondary image index.
            sec_key = np.concatenate(
                [np.full(len(x["dtScores"][0:maxDet]), i) for i, x in pairs])
            inds = np.lexsort((sec_key, -dtScores))
            dtm = np.concatenate([x["dtMatches"][:, 0:maxDet] for _, x in pairs], axis=1)[:, inds]
            dtIg = np.concatenate([x["dtIgnore"][:, 0:maxDet] for _, x in pairs], axis=1)[:, inds]
            gtIg = np.concatenate([x["gtIgnore"] for _, x in pairs])
            npig = np.count_nonzero(gtIg == 0)
            if npig == 0:
                continue
            tps = np.logical_and(dtm, np.logical_not(dtIg))
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
            tp_sum = np.cumsum(tps, axis=1).astype(float)
            fp_sum = np.cumsum(fps, axis=1).astype(float)
            for t in range(T):
                tp, fp = tp_sum[t], fp_sum[t]
                nd = len(tp)
                rc = tp / npig
                pr = tp / (fp + tp + np.spacing(1))
                # monotone envelope: pr[i] := max(pr[i:])
                pr = np.maximum.accumulate(pr[::-1])[::-1]
                q = np.zeros(R)
                ids = np.searchsorted(rc, p.recThrs, side="left")
                ok = ids < nd
                q[ok] = pr[ids[ok]]
                # NB: assign even when nd==0 (q stays all-zero). Stock
                # COCOeval does the same, and the distinction matters: a 0 row
                # counts in the mAP mean, a left-at--1 row is dropped from it.
                # Skipping this write inflates AP for bins where many classes
                # have GT but no detections at all.
                precision[t, :, k, a] = q

    out50_95, out50 = np.empty(A), np.empty(A)
    for a in range(A):
        s = precision[:, :, :, a]
        out50_95[a] = s[s > -1].mean() if (s > -1).any() else np.nan
        s50 = precision[0:1, :, :, a]
        out50[a] = s50[s50 > -1].mean() if (s50 > -1).any() else np.nan
    return out50_95, out50


def self_check(e):
    """Our trimmed accumulate must reproduce stock COCOeval.accumulate.

    A one-sided NaN (stock finite, ours NaN or vice-versa) is a real mismatch,
    not a no-op: max(worst, nan) would silently leave worst unchanged, so it is
    scored as +inf here instead of being swallowed.
    """
    e.accumulate()
    stock = e.eval["precision"]
    ours, _ = accumulate_ap(e, list(range(len(e._paramsEval.imgIds))))
    worst = 0.0
    for a in range(len(e.params.areaRng)):
        s = stock[:, :, :, a, 0]
        ref = s[s > -1].mean() if (s > -1).any() else np.nan
        both_nan = np.isnan(ref) and np.isnan(ours[a])
        one_nan = np.isnan(ref) != np.isnan(ours[a])
        if one_nan:
            worst = float("inf")
        elif not both_nan:
            worst = max(worst, abs(ref - ours[a]))
    return worst


def run_model(model, precision, gt_dataset, metrics_dir, n_boot, seed, do_self_check, height_only=False):
    with open(os.path.join(metrics_dir, f"pred_{model}_fp32.json"), encoding="utf-8") as f:
        dt_fp32 = json.load(f)
    with open(os.path.join(metrics_dir, f"pred_{model}_{precision}.json"), encoding="utf-8") as f:
        dt_q = json.load(f)

    # half-open [lo,hi) for our height bins (COCOeval is inclusive-both, so the
    # upper edge is shrunk by 1e-3 in area units — heights are fractional, see
    # eval_stratified.per_height_bin for why 1e-3 and not 0.5). COCO bins keep
    # the native inclusive convention to match the literature.
    h_rngs = [(lo * lo, BIG if np.isinf(hi) else hi * hi - 1e-3) for _, lo, hi in HEIGHT_BINS]
    c_rngs = [(lo, BIG if np.isinf(hi) else hi) for _, lo, hi in COCO_BINS]

    evals = {
        ("h", "fp32"): build_eval(gt_dataset, dt_fp32, h_rngs, "height2"),
        ("h", "q"): build_eval(gt_dataset, dt_q, h_rngs, "height2"),
    }
    # The COCO-bin CI (ci_SUR_cocoS_cocoL) is not used in the paper's headline
    # table, and each bootstrap resample re-accumulates over all 3067 images per
    # eval, so computing it doubles the runtime. Skip it unless asked.
    if not height_only:
        evals[("c", "fp32")] = build_eval(gt_dataset, dt_fp32, c_rngs, "true")
        evals[("c", "q")] = build_eval(gt_dataset, dt_q, c_rngs, "true")

    if do_self_check:
        d = self_check(evals[("h", "fp32")])
        print(f"  self-check vs stock accumulate: max|Δ| = {d:.2e} "
              f"{'OK' if d < 1e-9 else 'MISMATCH'}")
        if d >= 1e-9:
            raise SystemExit("custom accumulate does not match pycocotools — aborting")

    n_img = len(evals[("h", "fp32")]._paramsEval.imgIds)
    full = list(range(n_img))

    # point estimate on the full test set
    keys = ("h",) if height_only else ("h", "c")
    pt = {}
    for key in keys:
        ap_f, _ = accumulate_ap(evals[(key, "fp32")], full)
        ap_q, _ = accumulate_ap(evals[(key, "q")], full)
        pt[key] = (ap_f, ap_q)
    if height_only:  # placeholder so COCO fields stay well-formed
        n_cb = len(COCO_BINS)
        pt["c"] = (np.full(n_cb, np.nan), np.full(n_cb, np.nan))

    rng = np.random.default_rng(seed)
    boot = {"h_delta": [], "c_delta": []}
    pairs_to_boot = (("h", "h_delta"),) if height_only else (("h", "h_delta"), ("c", "c_delta"))
    for _ in range(n_boot):
        s = rng.choice(n_img, size=n_img, replace=True).tolist()
        for key, store in pairs_to_boot:
            ap_f, _ = accumulate_ap(evals[(key, "fp32")], s)
            ap_q, _ = accumulate_ap(evals[(key, "q")], s)
            boot[store].append(ap_f - ap_q)
    h_delta = np.array(boot["h_delta"])   # (n_boot, 5)
    c_delta = np.array(boot["c_delta"]) if boot["c_delta"] else np.full((n_boot, len(COCO_BINS)), np.nan)

    def ci(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return [float("nan")] * 3
        return [float(x) for x in np.percentile(v, [2.5, 50, 97.5])]

    def ratio_ci(num, den):
        # A ratio whose denominator can cross zero is non-regular: the percentile
        # interval here is only meaningful when the denominator is bounded away
        # from 0 (n_dropped small). When n_dropped is large the CI is NOT valid
        # coverage — report the DIFFERENCE (num-den) instead, which stays finite.
        # frac_denom_neg_num_pos = share of retained draws where numerator and
        # denominator disagree in sign (an unstable-ratio symptom), NOT a flip vs
        # the point estimate.
        m = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-4)
        dropped = int((~m).sum())
        if not m.any():
            return {"lo": float("nan"), "median": float("nan"), "hi": float("nan"),
                    "n_dropped": dropped, "n_kept": 0, "frac_opposite_sign": float("nan")}
        r = num[m] / den[m]
        lo, med, hi = ci(r)
        return {"lo": lo, "median": med, "hi": hi, "n_dropped": dropped, "n_kept": int(m.sum()),
                "frac_opposite_sign": float((r < 0).mean())}

    def safe_ratio(a, b):
        return float(a / b) if abs(b) > 1e-9 else None  # None -> valid JSON, not Infinity

    hb = [b[0] for b in HEIGHT_BINS]
    cb = [b[0] for b in COCO_BINS]
    iXS, iS, iXL = hb.index("XS"), hb.index("S"), hb.index("XL")
    iCS, iCL = cb.index("small"), cb.index("large")

    result = {
        "model": model,
        "precision": precision,
        "n_boot": n_boot,
        "n_images": n_img,
        "point": {
            "delta_height": {b: float(pt["h"][0][i] - pt["h"][1][i]) for i, b in enumerate(hb)},
            "delta_coco": {b: float(pt["c"][0][i] - pt["c"][1][i]) for i, b in enumerate(cb)},
            "ap_fp32_height": {b: float(pt["h"][0][i]) for i, b in enumerate(hb)},
            "ap_fp32_coco": {b: float(pt["c"][0][i]) for i, b in enumerate(cb)},
            "SUR_XS_XL": safe_ratio(pt["h"][0][iXS] - pt["h"][1][iXS],
                                    pt["h"][0][iXL] - pt["h"][1][iXL]),
            "SUR_S_XL": safe_ratio(pt["h"][0][iS] - pt["h"][1][iS],
                                   pt["h"][0][iXL] - pt["h"][1][iXL]),
            "SUR_cocoS_cocoL": safe_ratio(pt["c"][0][iCS] - pt["c"][1][iCS],
                                          pt["c"][0][iCL] - pt["c"][1][iCL]),
        },
        "ci_delta_height": {b: ci(h_delta[:, i]) for i, b in enumerate(hb)},
        "ci_delta_coco": {b: ci(c_delta[:, i]) for i, b in enumerate(cb)},
        "ci_SUR_XS_XL": ratio_ci(h_delta[:, iXS], h_delta[:, iXL]),
        "ci_SUR_S_XL": ratio_ci(h_delta[:, iS], h_delta[:, iXL]),
        "ci_SUR_cocoS_cocoL": ratio_ci(c_delta[:, iCS], c_delta[:, iCL]),
        # difference-based alternative: finite even when the denominator -> 0
        "ci_DIFF_S_minus_XL": ci(h_delta[:, iS] - h_delta[:, iXL]),
        "ci_DIFF_XS_minus_XL": ci(h_delta[:, iXS] - h_delta[:, iXL]),
        "ci_DIFF_cocoS_minus_cocoL": ci(c_delta[:, iCS] - c_delta[:, iCL]),
    }
    return result


def main(models, precision, gt_path, metrics_dir, n_boot, seed, out_path, do_self_check, height_only=False):
    with open(gt_path, encoding="utf-8") as f:
        gt_dataset = json.load(f)

    all_res = {}
    for m in models:
        print(f"\n### {m} ({precision}) — {n_boot} bootstrap resamples")
        r = run_model(m, precision, gt_dataset, metrics_dir, n_boot, seed, do_self_check, height_only)
        all_res[m] = r
        p = r["point"]
        fmt = lambda v: "None" if v is None else f"{v:+.2f}"
        print(f"  point   SUR_XS/XL={fmt(p['SUR_XS_XL'])}  SUR_S/XL={fmt(p['SUR_S_XL'])}  "
              f"SUR_cocoS/L={fmt(p['SUR_cocoS_cocoL'])}")
        for k in ("ci_SUR_XS_XL", "ci_SUR_S_XL", "ci_SUR_cocoS_cocoL"):
            c = r[k]
            print(f"  {k:22} median={c['median']:+.2f}  CI95=[{c['lo']:+.2f}, {c['hi']:+.2f}]  "
                  f"opp-sign={c['frac_opposite_sign']:.1%}  dropped={c['n_dropped']}")
        # DIFF is the headline stat — finite even when Δ_XL -> 0, so no truncation
        d = r["ci_DIFF_S_minus_XL"]
        print(f"  DIFF Δ_S-Δ_XL (HEADLINE) median={d[1]:+.4f}  CI95=[{d[0]:+.4f}, {d[2]:+.4f}]  "
              f"{'EXCLUDES 0' if (d[0] > 0 or d[2] < 0) else 'includes 0'}")
        do_self_check = False  # once is enough

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precision", default="int8_ptq")
    ap.add_argument("--gt", default=os.path.join(REPO_ROOT, "metrics", "gt_test.json"))
    ap.add_argument("--metrics-dir", default=os.path.join(REPO_ROOT, "metrics"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "metrics", "sur_bootstrap.json"))
    ap.add_argument("--no-self-check", action="store_true")
    ap.add_argument("--height-only", action="store_true", help="skip COCO-bin evals (2x faster; paper uses S/XL height CI)")
    args = ap.parse_args()
    main(args.models, args.precision, args.gt, args.metrics_dir, args.n_boot, args.seed,
         args.out, not args.no_self_check, args.height_only)
