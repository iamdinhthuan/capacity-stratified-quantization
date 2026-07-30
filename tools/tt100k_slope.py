#!/usr/bin/env python3
"""The capacity slope on TT100K, where training is controlled.

Every COCO checkpoint is somebody else's training run, so a slope measured
there could in principle reflect how the release recipes were tuned across
model sizes rather than a property of quantization. TT100K removes that
possibility: all five detectors were trained here under one recipe with
only `model` changed. If the slope survives a dataset change, a class-count
change (45 vs 80), a resolution change (1280 vs 640) and a switch to
controlled training, it is a property of capacity and not of the release
pipeline.

Only one family is available with a full sweep, so no fixed effects are
needed and the permutation is the exact enumeration of all 5! orderings
(minimum attainable p = 1/120).
"""
import itertools, json, os
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(R, "metrics")
MODELS = {"yolo11n": 2.6, "yolo11s": 9.5, "yolo11m": 20.1, "yolo11l": 25.3, "yolo11x": 57.0}


def load(m, prec):
    return json.load(open(os.path.join(M, f"{m}_{prec}.json")))


def gap(m, prec, small, large, field):
    f, q = load(m, "fp32"), load(m, prec)
    d = lambda j, b: j[field][b]["mAP50-95"]
    return (d(f, small) - d(q, small)) - (d(f, large) - d(q, large))


def exact_slope_test(x, y):
    """Exact one-sided permutation over all orderings of y."""
    xc = x - x.mean()
    obs = float((xc * y).sum() / (xc ** 2).sum())
    stats = [float((xc * np.array(p)).sum() / (xc ** 2).sum())
             for p in itertools.permutations(y)]
    p = sum(1 for s in stats if s >= obs) / len(stats)
    return obs, p, len(stats)


out = {}
x = np.array([np.log10(MODELS[m]) for m in MODELS])
for prec in ("int8_ptq", "fp8"):
    for tag, (sm, lg, field) in {
        "height_S_minus_XL": ("S", "XL", "height_bin_ap"),
        "coco_small_minus_large": ("small", "large", "coco_bin_ap"),
    }.items():
        y = np.array([gap(m, prec, sm, lg, field) for m in MODELS])
        b, p, n = exact_slope_test(x, y)
        out[f"{prec}_{tag}"] = {
            "slope_per_decade": b, "p_exact_one_sided": p, "n_permutations": n,
            "diff": {m: round(float(v), 5) for m, v in zip(MODELS, y)}}
        print(f"{prec:9s} {tag:24s} slope={b:+.4f}/decade  p={p:.4f} (exact, {n})")
        print(f"          {'  '.join(f'{m[-1]}:{v:+.4f}' for m, v in zip(MODELS, y))}")

# Decomposition on the COCO-style strata, INT8.
for band in ("small", "large"):
    y = np.array([load(m, "fp32")["coco_bin_ap"][band]["mAP50-95"]
                  - load(m, "int8_ptq")["coco_bin_ap"][band]["mAP50-95"] for m in MODELS])
    b, p, _ = exact_slope_test(x, y)
    out[f"int8_delta_{band}"] = {"slope_per_decade": b, "p_exact_one_sided_increasing": p,
                                 "values": {m: round(float(v), 5) for m, v in zip(MODELS, y)}}
    print(f"int8      dAP_{band:20s} slope={b:+.4f}/decade  p(incr)={p:.4f}")

json.dump(out, open(os.path.join(M, "tt100k_capacity_slope.json"), "w"), indent=1)
print("->", os.path.join(M, "tt100k_capacity_slope.json"))
