#!/usr/bin/env python3
"""Robustness of the continuous capacity slope.

A single steep point (YOLO26n, DIFF = -0.089) could in principle carry an
OLS slope by itself, and the whole claim rests on this statistic, so it is
stress-tested four ways before being reported: drop the nano rung, switch
to a rank statistic that cannot be levered by one extreme value, drop each
family in turn, and drop each model in turn.
"""
import json, os
import numpy as np
from coco_slope_test import PARAMS, FAM, SETS, diff, slope

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(R, "metrics", "coco_5090")


def perm_p(x, y, fi, nf, stat, nperm=20000, seed=7):
    obs = stat(x, y, fi, nf)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(nperm):
        yp = y.copy()
        for f in range(nf):
            k = np.where(fi == f)[0]
            yp[k] = rng.permutation(y[k])
        if stat(x, yp, fi, nf) >= obs:
            ge += 1
    return obs, (1 + ge) / (1 + nperm)


def ols_stat(x, y, fi, nf):
    return slope(x, y, fi, nf)[0]


def rank_stat(x, y, fi, nf):
    """Mean within-family Kendall tau -- immune to a single extreme value."""
    ts = []
    for f in range(nf):
        k = np.where(fi == f)[0]
        xs, ys = x[k], y[k]
        c = d = 0
        for i in range(len(k)):
            for j in range(i + 1, len(k)):
                s = np.sign(xs[i] - xs[j]) * np.sign(ys[i] - ys[j])
                c += s > 0
                d += s < 0
        ts.append((c - d) / max(c + d, 1))
    return float(np.mean(ts))


def prep(models):
    models = sorted(models, key=lambda m: (FAM[m], PARAMS[m]))
    fams = sorted({FAM[m] for m in models})
    return (np.array([np.log10(PARAMS[m]) for m in models]),
            np.array([diff(m) for m in models]),
            np.array([fams.index(FAM[m]) for m in models]), len(fams), models)


out = {}
base = SETS["cnn15"]

for label, ms in [("cnn15", base),
                  ("registered10", SETS["registered10"]),
                  ("confirm_cnn10", SETS["confirm_cnn10"]),
                  ("cnn15_no_nano", [m for m in base if not m.endswith("n")]),
                  ("cnn15_no_yolo26n", [m for m in base if m != "yolo26n"])]:
    x, y, fi, nf, mm = prep(ms)
    b, pb = perm_p(x, y, fi, nf, ols_stat)
    t, pt = perm_p(x, y, fi, nf, rank_stat)
    out[label] = {"n": len(mm), "slope": b, "p_slope": pb,
                  "mean_within_family_tau": t, "p_tau": pt}
    print(f"{label:18s} n={len(mm):2d}  slope={b:+.4f} p={pb:.4f}   "
          f"within-family tau={t:+.3f} p={pt:.4f}")

print("\nleave-one-family-out (CNN):")
lofo = {}
for drop in ("yolo11", "yolov8", "yolo26"):
    ms = [m for m in base if FAM[m] != drop]
    x, y, fi, nf, mm = prep(ms)
    b, pb = perm_p(x, y, fi, nf, ols_stat, nperm=10000)
    lofo[drop] = {"slope": b, "p": pb}
    print(f"  without {drop:8s}: slope={b:+.4f}  p={pb:.4f}")
out["leave_one_family_out"] = lofo

print("\nleave-one-model-out (CNN), worst cases:")
loo = {}
for drop in base:
    ms = [m for m in base if m != drop]
    x, y, fi, nf, mm = prep(ms)
    b, pb = perm_p(x, y, fi, nf, ols_stat, nperm=10000)
    loo[drop] = {"slope": b, "p": pb}
worst = sorted(loo.items(), key=lambda kv: -kv[1]["p"])[:3]
for k, v in worst:
    print(f"  without {k:9s}: slope={v['slope']:+.4f}  p={v['p']:.4f}")
print(f"  max p over all 15 deletions = {max(v['p'] for v in loo.values()):.4f}")
out["leave_one_model_out"] = loo
out["loo_max_p"] = max(v["p"] for v in loo.values())

json.dump(out, open(os.path.join(M, "capacity_slope_robust.json"), "w"), indent=1)
print("->", os.path.join(M, "capacity_slope_robust.json"))
