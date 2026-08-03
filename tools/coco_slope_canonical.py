"""Every capacity-slope number quoted in the article, from one script and one seed.

Reads the committed per-model metric files, fits DIFF_i = a_family + b*log10(params)
by OLS with family fixed effects, calibrates b by permutation within each family
(20,000 draws) and its CI by a bootstrap within each family (B=2,000), and writes
metrics/coco_5090/capacity_slope_canonical.json. Table 5, Appendix F and Appendix H
quote this file; earlier drafts mixed two runs of the same fit that differed in the
last digit of the bootstrap CI.
"""
import json, os, numpy as np

M = os.path.join(os.path.dirname(__file__), "..", "metrics", "coco_5090")
PAR = {"yolo11n":2.6,"yolo11s":9.4,"yolo11m":20.1,"yolo11l":25.3,"yolo11x":56.9,
       "yolov8n":3.2,"yolov8s":11.2,"yolov8m":25.9,"yolov8l":43.7,"yolov8x":68.2,
       "yolo26n":2.4,"yolo26s":9.5,"yolo26m":20.4,"yolo26l":24.8,"yolo26x":55.7,
       "dfine_n":4.0,"dfine_s":10.0,"dfine_m":19.0,"dfine_l":31.0,"dfine_x":62.0}
CNN = [m for m in PAR if not m.startswith("dfine")]

def stats(model, prec):
    return json.load(open(os.path.join(M, f"{model}_{prec}.json")))["stats"]

def series(models, prec, kind):
    out = {}
    for m in models:
        r, q = stats(m, "fp32"), stats(m, prec)
        dS, dL = r["AP_small"]-q["AP_small"], r["AP_large"]-q["AP_large"]
        out[m] = {"DIFF": dS-dL, "large": dL, "small": dS, "agg": r["AP"]-q["AP"]}[kind]
    return out

def fam(m):
    for f in ("yolo11","yolov8","yolo26","dfine"):
        if m.startswith(f): return f
    raise ValueError(m)

def fit(d, nperm=20000, nboot=2000, seed=0):
    ms = sorted(d); fams = sorted({fam(m) for m in ms})
    X = np.zeros((len(ms), 1+len(fams))); y = np.array([d[m] for m in ms])
    for i, m in enumerate(ms):
        X[i,0] = np.log10(PAR[m]); X[i, 1+fams.index(fam(m))] = 1.0
    b0 = float(np.linalg.lstsq(X, y, rcond=None)[0][0])
    rng = np.random.default_rng(seed)
    idx = {f:[i for i,m in enumerate(ms) if fam(m)==f] for f in fams}
    ge = ab = 0
    for _ in range(nperm):
        yp = y.copy()
        for f, ii in idx.items(): yp[ii] = y[rng.permutation(ii)]
        b = np.linalg.lstsq(X, yp, rcond=None)[0][0]
        ge += b >= b0; ab += abs(b) >= abs(b0)
    p_pos = (ge+1)/(nperm+1)                       # one-sided toward +
    p_dir = min(p_pos, ( nperm-ge +1)/(nperm+1))   # one-sided toward the observed sign
    p_two = (ab+1)/(nperm+1)
    rng2 = np.random.default_rng(seed+1); draws = []
    for _ in range(nboot):
        sel = []
        for f, ii in idx.items(): sel += list(rng2.choice(ii, len(ii), replace=True))
        draws.append(float(np.linalg.lstsq(X[sel], y[sel], rcond=None)[0][0]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(n=len(ms), slope=b0, ci95=[float(lo), float(hi)],
                p_one_sided_positive=p_pos, p_one_sided_directional=p_dir, p_two_sided=p_two)

no_nano = [m for m in CNN if not m.endswith("n")]
out = {"seed": 0, "n_perm": 20000, "n_boot": 2000, "params_M": PAR, "rows": {
    "registered_primary_int8_DIFF": fit(series([m for m in PAR if fam(m) in ("yolov8","dfine")], "int8", "DIFF")),
    "convolutional15_int8_DIFF":    fit(series(CNN, "int8", "DIFF")),
    "registered_plus_exploratory_int8_DIFF": fit(series(list(PAR), "int8", "DIFF")),
    "convolutional15_int8_dAP_large": fit(series(CNN, "int8", "large")),
    "convolutional15_int8_dAP_small": fit(series(CNN, "int8", "small")),
    "convolutional15_int8_dAP_aggregate": fit(series(CNN, "int8", "agg")),
    "convolutional15_fp8_DIFF":     fit(series(CNN, "fp8", "DIFF")),
    "no_nano12_int8_DIFF":          fit(series(no_nano, "int8", "DIFF")),
    "convolutional15_int8ent_DIFF": fit(series(CNN, "int8ent", "DIFF")),
    "no_nano12_int8ent_DIFF":       fit(series(no_nano, "int8ent", "DIFF")),
}}
dst = os.path.join(M, "capacity_slope_canonical.json")
json.dump(out, open(dst, "w"), indent=1, sort_keys=True)
for k, v in out["rows"].items():
    print(f"{k:<42} n={v['n']:2d} b={v['slope']:+.4f} CI=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] "
          f"p_dir={v['p_one_sided_directional']:.4f} p_two={v['p_two_sided']:.4f}")
print("->", dst)
