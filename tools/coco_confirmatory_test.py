"""Pre-registered confirmatory analysis (PREREGISTRATION_coco_confirmatory.md).

H1 (one-sided): mean DIFF(>=20M) > mean DIFF(<20M) under INT8, across the
confirmatory set (YOLOv8 n/s/m/l/x + D-FINE N/S/M/L/X; YOLO26 secondary).

Primary test: exact permutation stratified by family — within each family,
relabel which models are 'low'/'high' in every way that preserves that
family's group sizes; statistic = pooled mean(high) - mean(low).
Robustness: one-sided Welch's t. Also reports Kendall's tau of DIFF vs
log(params) and echoes per-model DIFF for the forest plot.

Run AFTER the full confirmatory set is evaluated (no-peek rule).

Usage:
    python tools/coco_confirmatory_test.py \
        --spec yolov8n:3.2:low yolov8s:11.2:low yolov8m:25.9:high \
               yolov8l:43.7:high yolov8x:68.2:high \
               dfine_n:4:low dfine_s:10:low dfine_m:19:low \
               dfine_l:31:high dfine_x:62:high
"""
import argparse
import itertools
import json
import math
import os

import numpy as np

from coco_common import PILOT_METRICS


def diff_of(model, quant="int8", ref="fp32"):
    def stats(p):
        with open(os.path.join(PILOT_METRICS, f"{model}_{p}.json"), encoding="utf-8") as f:
            return json.load(f)["stats"]
    r, q = stats(ref), stats(quant)
    return (r["AP_small"] - q["AP_small"]) - (r["AP_large"] - q["AP_large"])


def family_of(model):
    for fam in ("yolov8", "yolo26", "yolo11", "dfine"):
        if model.startswith(fam):
            return fam
    return model.rstrip("nsmlx")


def exact_stratified_perm(entries):
    """entries: list of (family, diff, is_high). Returns (observed, p_one_sided, n_perms)."""
    obs_high = [d for _, d, h in entries if h]
    obs_low = [d for _, d, h in entries if not h]
    observed = np.mean(obs_high) - np.mean(obs_low)

    fams = {}
    for fam, d, h in entries:
        fams.setdefault(fam, {"diffs": [], "n_high": 0})
        fams[fam]["diffs"].append(d)
        fams[fam]["n_high"] += int(h)

    per_fam_choices = []
    for fam, info in fams.items():
        idx = range(len(info["diffs"]))
        per_fam_choices.append([set(c) for c in itertools.combinations(idx, info["n_high"])])

    count_ge, total = 0, 0
    for combo in itertools.product(*per_fam_choices):
        hi, lo = [], []
        for (fam, info), chosen in zip(fams.items(), combo):
            for i, d in enumerate(info["diffs"]):
                (hi if i in chosen else lo).append(d)
        stat = np.mean(hi) - np.mean(lo)
        total += 1
        if stat >= observed - 1e-12:
            count_ge += 1
    return observed, count_ge / total, total


def welch_one_sided(high, low):
    h, l = np.array(high), np.array(low)
    vh, vl = h.var(ddof=1) / len(h), l.var(ddof=1) / len(l)
    t = (h.mean() - l.mean()) / math.sqrt(vh + vl)
    df = (vh + vl) ** 2 / (vh ** 2 / (len(h) - 1) + vl ** 2 / (len(l) - 1))
    from scipy import stats as st
    return t, df, float(1 - st.t.cdf(t, df))


def kendall_tau(diffs, params):
    from scipy import stats as st
    tau, p = st.kendalltau(np.log(params), diffs)
    return float(tau), float(p)


def main(spec, quant, out_path):
    entries, params, diffs = [], [], []
    print(f"{'model':10} {'params':>7} {'group':>5} {'DIFF':>9}")
    for item in spec:
        model, p, grp = item.split(":")
        d = diff_of(model, quant)
        entries.append((family_of(model), d, grp == "high"))
        params.append(float(p)); diffs.append(d)
        print(f"{model:10} {p:>6}M {grp:>5} {d:+.4f}")

    observed, p_perm, n_perms = exact_stratified_perm(entries)
    hi = [d for _, d, h in entries if h]
    lo = [d for _, d, h in entries if not h]
    t, df, p_welch = welch_one_sided(hi, lo)
    tau, p_tau = kendall_tau(diffs, params)

    result = {
        "quant": quant,
        "observed_mean_high_minus_low": observed,
        "p_perm_one_sided": p_perm,
        "n_permutations": n_perms,
        "welch_t": t, "welch_df": df, "p_welch_one_sided": p_welch,
        "kendall_tau_vs_logparams": tau, "p_kendall_two_sided": p_tau,
        "per_model": [{"model": s.split(":")[0], "params_M": float(s.split(":")[1]),
                        "group": s.split(":")[2], "DIFF": d}
                       for s, d in zip(spec, diffs)],
    }
    print(f"\nH1 mean(high)-mean(low) = {observed:+.4f}")
    print(f"exact stratified permutation: p = {p_perm:.4f} ({n_perms} rearrangements)")
    print(f"Welch one-sided: t={t:.3f}, df={df:.1f}, p={p_welch:.4f}")
    print(f"Kendall tau vs log(params): {tau:+.3f} (p={p_tau:.4f})")
    verdict = "H1 SUPPORTED" if p_perm < 0.05 else "H1 NOT SUPPORTED"
    print(f"=> {verdict} at alpha=0.05 (pre-registered)")
    result["verdict"] = verdict
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True, help="model:paramsM:low|high")
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--out", default=os.path.join(PILOT_METRICS, "confirmatory_test.json"))
    args = ap.parse_args()
    main(args.spec, args.quant, args.out)
