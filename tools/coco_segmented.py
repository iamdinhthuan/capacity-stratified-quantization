#!/usr/bin/env python3
"""Segmented (broken-stick) regression of DIFF on log-capacity.

The analysis plan promised a breakpoint estimate for the capacity trend.
Kendall's tau answers "is there a monotone trend"; it does not answer the
question the paper actually poses, which is *where* the small-minus-large
gap changes character. This fits Muggeo's segmented model

    DIFF(x) = b0 + b1*x + b2*(x - psi)_+ ,    x = log10(params in M)

by profiling the likelihood over a dense grid of psi (exact for a fixed
breakpoint, since the model is linear in the other coefficients), and puts
a nonparametric bootstrap interval on psi by resampling models within
family. Family is absorbed as a fixed effect so that a lineage's mean
level cannot masquerade as a capacity effect.

The honest expected outcome with 15 points is a wide interval. That is the
point: it converts "the crossing is not localisable" from an assertion
into a measured width.
"""
import json, itertools, os, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(R, "metrics", "coco_5090")

# Official parameter counts (millions), from the checkpoints evaluated.
PARAMS = {
    "yolo11n": 2.6, "yolo11s": 9.4, "yolo11m": 20.1, "yolo11l": 25.3, "yolo11x": 56.9,
    "yolov8n": 3.2, "yolov8s": 11.2, "yolov8m": 25.9, "yolov8l": 43.7, "yolov8x": 68.2,
    "yolo26n": 2.4, "yolo26s": 9.5, "yolo26m": 20.4, "yolo26l": 24.8, "yolo26x": 55.7,
}
FAMILY = {m: ("yolo11" if m.startswith("yolo11") else
              "yolov8" if m.startswith("yolov8") else "yolo26") for m in PARAMS}


def diff(model, prec="int8"):
    """DIFF = dAP_small - dAP_large, both as FP32 minus quantized."""
    f = json.load(open(os.path.join(M, f"{model}_fp32.json")))["stats"]
    q = json.load(open(os.path.join(M, f"{model}_{prec}.json")))["stats"]
    return (f["AP_small"] - q["AP_small"]) - (f["AP_large"] - q["AP_large"])


def design(x, psi, fam_idx, n_fam):
    """[intercept-per-family | x | (x-psi)_+]"""
    n = len(x)
    Z = np.zeros((n, n_fam + 2))
    Z[np.arange(n), fam_idx] = 1.0
    Z[:, n_fam] = x
    Z[:, n_fam + 1] = np.maximum(x - psi, 0.0)
    return Z


def fit(x, y, fam_idx, n_fam, grid):
    best = None
    for psi in grid:
        Z = design(x, psi, fam_idx, n_fam)
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        rss = float(((y - Z @ beta) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, psi, beta)
    return best


def main():
    models = sorted(PARAMS, key=lambda m: (FAMILY[m], PARAMS[m]))
    x = np.array([np.log10(PARAMS[m]) for m in models])
    y = np.array([diff(m) for m in models])
    fams = sorted(set(FAMILY.values()))
    fam_idx = np.array([fams.index(FAMILY[m]) for m in models])
    nf = len(fams)

    # Interior grid only: a breakpoint outside the data is not a breakpoint.
    grid = np.linspace(x.min() + 0.12, x.max() - 0.12, 400)
    rss1, psi_hat, beta = fit(x, y, fam_idx, nf, grid)

    # Null: single straight line (no break), same family intercepts.
    Z0 = np.column_stack([np.eye(nf)[fam_idx], x])
    b0, *_ = np.linalg.lstsq(Z0, y, rcond=None)
    rss0 = float(((y - Z0 @ b0) ** 2).sum())

    # Davies' problem: psi is unidentified under the null, so the usual F
    # reference distribution does not apply. Calibrate by permuting y within
    # family, which preserves each lineage's level and spread but destroys
    # any capacity ordering.
    rng = np.random.default_rng(20260730)
    obs_F = (rss0 - rss1) / (rss1 / max(len(x) - nf - 2, 1))
    null_F = []
    for _ in range(5000):
        yp = y.copy()
        for f in range(nf):
            k = np.where(fam_idx == f)[0]
            yp[k] = rng.permutation(y[k])
        r1, _, _ = fit(x, yp, fam_idx, nf, grid)
        Zn = np.column_stack([np.eye(nf)[fam_idx], x])
        bn, *_ = np.linalg.lstsq(Zn, yp, rcond=None)
        r0 = float(((yp - Zn @ bn) ** 2).sum())
        null_F.append((r0 - r1) / (r1 / max(len(x) - nf - 2, 1)))
    p_break = (1 + sum(1 for v in null_F if v >= obs_F)) / (1 + len(null_F))

    # Bootstrap interval on psi: resample models with replacement *within*
    # family, so every draw keeps all three lineages represented.
    psis = []
    for _ in range(4000):
        idx = np.concatenate([rng.choice(np.where(fam_idx == f)[0],
                                         size=(fam_idx == f).sum(), replace=True)
                              for f in range(nf)])
        xb, yb, fb = x[idx], y[idx], fam_idx[idx]
        if len(np.unique(xb)) < 4:
            continue
        try:
            _, p, _ = fit(xb, yb, fb, nf, grid)
            psis.append(p)
        except np.linalg.LinAlgError:
            continue
    psis = np.array(psis)
    lo, hi = np.percentile(psis, [2.5, 97.5])

    # Zero-crossing of the family-averaged fitted curve, for comparison with
    # the sign-flip window quoted from the raw ladders.
    xs = np.linspace(x.min(), x.max(), 2000)
    Zs = design(xs, psi_hat, np.zeros(len(xs), int), nf)
    Zs[:, :nf] = beta[:nf].mean()  # family-averaged intercept
    Zs[:, :nf] = 0.0
    curve = beta[:nf].mean() + beta[nf] * xs + beta[nf + 1] * np.maximum(xs - psi_hat, 0)
    sign = np.where(np.diff(np.sign(curve)) != 0)[0]
    cross = float(10 ** xs[sign[0]]) if len(sign) else None

    out = {
        "models": models,
        "log10_params": x.round(4).tolist(),
        "diff_int8": y.round(5).tolist(),
        "breakpoint_log10M": float(psi_hat),
        "breakpoint_M": float(10 ** psi_hat),
        "breakpoint_ci95_M": [float(10 ** lo), float(10 ** hi)],
        "ci_width_ratio": float(10 ** hi / 10 ** lo),
        "slope_before": float(beta[nf]),
        "slope_after": float(beta[nf] + beta[nf + 1]),
        "rss_line": rss0, "rss_segmented": rss1,
        "F_obs": float(obs_F),
        "p_break_permutation": float(p_break),
        "zero_crossing_M": cross,
        "n_bootstrap": int(len(psis)),
        "note": ("Family fixed effects absorbed; breakpoint profiled over an "
                 "interior grid; Davies-safe p from within-family permutation "
                 "(5000); psi CI from within-family bootstrap."),
    }
    dst = os.path.join(R, "metrics", "coco_5090", "segmented.json")
    json.dump(out, open(dst, "w"), indent=1)
    for m, xi, yi in zip(models, x, y):
        print(f"{m:9s} log10P={xi:.3f} DIFF={yi:+.4f}")
    print(f"\nbreakpoint  = {out['breakpoint_M']:.1f}M  "
          f"95% CI [{out['breakpoint_ci95_M'][0]:.1f}, {out['breakpoint_ci95_M'][1]:.1f}]M "
          f"(x{out['ci_width_ratio']:.1f} wide)")
    print(f"slopes      = {out['slope_before']:+.4f} -> {out['slope_after']:+.4f} per decade")
    print(f"break vs line: F={obs_F:.2f}, permutation p={p_break:.3f}")
    print(f"zero crossing of fitted curve: "
          f"{cross:.1f}M" if cross else "no zero crossing in range")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
