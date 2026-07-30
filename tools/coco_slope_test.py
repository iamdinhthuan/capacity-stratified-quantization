#!/usr/bin/env python3
"""Continuous capacity slope: the test the dichotomy could not run.

The registered endpoint split models at 20M and compared two group means.
Dichotomising a continuous predictor discards the ordering inside each
group and is the least powerful way to ask whether DIFF tracks capacity.
This fits the continuous alternative

    DIFF_i = alpha_{family(i)} + beta * log10(params_i) + e_i

and tests beta > 0 by permuting DIFF *within family* -- the same
randomisation the registered test used, so the two are calibrated against
the same null and differ only in the statistic. Family fixed effects mean
beta is identified from capacity ordering inside lineages, never from the
level difference between them.

Reported for three nested sets: the ten confirmatory models (the
registered set), the fifteen convolutional models, and all twenty
including D-FINE, so the reader can see exactly which models carry it.
"""
import json, os
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(R, "metrics", "coco_5090")

PARAMS = {
    "yolo11n": 2.6, "yolo11s": 9.4, "yolo11m": 20.1, "yolo11l": 25.3, "yolo11x": 56.9,
    "yolov8n": 3.2, "yolov8s": 11.2, "yolov8m": 25.9, "yolov8l": 43.7, "yolov8x": 68.2,
    "yolo26n": 2.4, "yolo26s": 9.5, "yolo26m": 20.4, "yolo26l": 24.8, "yolo26x": 55.7,
    "dfine_n": 4.0, "dfine_s": 10.0, "dfine_m": 19.0, "dfine_l": 31.0, "dfine_x": 62.0,
}
FAM = {m: m.rstrip("nsmlx").rstrip("_") for m in PARAMS}

# The pre-registration names YOLOv8 and D-FINE as the primary set and YOLO26
# as secondary; "registered10" must therefore include the family that broke
# the frame, or the continuous statistic would be scored on a set chosen
# after seeing which family failed.
SETS = {
    "registered10":  [m for m in PARAMS if m.startswith(("yolov8", "dfine"))],
    "cnn15":         [m for m in PARAMS if not m.startswith("dfine")],
    "all20":         list(PARAMS),
    "confirm_cnn10": [m for m in PARAMS if m.startswith(("yolov8", "yolo26"))],
}


def diff(model, prec="int8"):
    f = json.load(open(os.path.join(M, f"{model}_fp32.json")))["stats"]
    q = json.load(open(os.path.join(M, f"{model}_{prec}.json")))["stats"]
    return (f["AP_small"] - q["AP_small"]) - (f["AP_large"] - q["AP_large"])


def slope(x, y, fam_idx, nf):
    Z = np.column_stack([np.eye(nf)[fam_idx], x])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    dof = len(x) - nf - 1
    s2 = float((resid ** 2).sum()) / dof
    xc = x - np.eye(nf)[fam_idx] @ np.array(
        [x[fam_idx == f].mean() for f in range(nf)])
    se = float(np.sqrt(s2 / (xc ** 2).sum()))
    return float(beta[-1]), se, dof


def run(models, prec="int8", nperm=20000, seed=20260730):
    models = sorted(models, key=lambda m: (FAM[m], PARAMS[m]))
    fams = sorted({FAM[m] for m in models})
    x = np.array([np.log10(PARAMS[m]) for m in models])
    y = np.array([diff(m, prec) for m in models])
    fi = np.array([fams.index(FAM[m]) for m in models])
    nf = len(fams)

    b, se, dof = slope(x, y, fi, nf)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(nperm):
        yp = y.copy()
        for f in range(nf):
            k = np.where(fi == f)[0]
            yp[k] = rng.permutation(y[k])
        if slope(x, yp, fi, nf)[0] >= b:
            ge += 1
    p1 = (1 + ge) / (1 + nperm)

    # Bootstrap CI on beta, resampling models within family.
    bs = []
    for _ in range(4000):
        idx = np.concatenate([rng.choice(np.where(fi == f)[0], (fi == f).sum(), True)
                              for f in range(nf)])
        if len(np.unique(x[idx])) < nf + 2:
            continue
        try:
            bs.append(slope(x[idx], y[idx], fi[idx], nf)[0])
        except np.linalg.LinAlgError:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": len(models), "families": fams, "slope_per_decade": b,
            "se": se, "dof": dof, "ci95": [float(lo), float(hi)],
            "p_one_sided_permutation": p1, "n_perm": nperm,
            "diffs": {m: round(float(v), 5) for m, v in zip(models, y)}}


if __name__ == "__main__":
    out = {}
    for prec in ("int8", "fp8"):
        for name, ms in SETS.items():
            key = f"{name}_{prec}"
            r = run(ms, prec)
            out[key] = r
            print(f"{key:22s} n={r['n']:2d}  beta={r['slope_per_decade']:+.4f} "
                  f"AP/decade  95% CI [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  "
                  f"p(one-sided)={r['p_one_sided_permutation']:.4f}")
    dst = os.path.join(M, "capacity_slope.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("->", dst)
