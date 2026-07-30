"""Is the size-unfairness effect driven by capacity, or by small-object headroom?

Reviewer B#2 / D2. The obvious referee question about "model capacity decides
who degrades first" is: *"small models have SUR<1 only because they never
detected small signs at FP32 — there was nothing left to lose. Is that capacity,
or just a floor effect?"* Unanswered, that is a reject.

So we regress the unfairness statistic against competing explanatory variables
and let the data pick:

  params        -- the "capacity" story as originally framed
  AP_small(FP32)-- the "headroom / floor" story (several bin definitions)
  mAP(FP32)     -- generic "the model is just better" confound
  Δ_XL          -- sanity check on the RATIO itself: if SUR tracks 1/Δ_XL, the
                   ordering is an artifact of the denominator, not of small signs

Reported per candidate: Spearman (rank, robust, what we trust at this n) and
Pearson/R² (shape, reported for completeness). With n<=7 models these are
descriptive, not confirmatory — the script prints that caveat next to the p.

Both a ratio target (SUR) and a difference target (Δ_small − Δ_large) are
analysed; the difference stays finite when Δ_XL → 0, so if the two disagree,
believe the difference.

Usage:
    python tools/mechanism_analysis.py --models yolo26n yolo11s yolov8s yolo11m \
        --precision int8_ptq --out metrics/mechanism.json
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

from common import REPO_ROOT


def count_params(model):
    """Parameter count of the actually-trained network (nc=221 head included)."""
    from ultralytics import YOLO
    p = os.path.join(REPO_ROOT, "runs", "detect", "runs", f"{model}_fp32", "weights", "best.pt")
    if not os.path.exists(p):
        return float("nan")
    y = YOLO(p)
    return sum(q.numel() for q in y.model.parameters()) / 1e6


def load_rows(models, precision, metrics_dir, boot_path):
    boot = {}
    if boot_path and os.path.exists(boot_path):
        with open(boot_path, encoding="utf-8") as f:
            boot = json.load(f)

    rows = []
    for m in models:
        f32p = os.path.join(metrics_dir, f"{m}_fp32.json")
        qp = os.path.join(metrics_dir, f"{m}_{precision}.json")
        if not (os.path.exists(f32p) and os.path.exists(qp)):
            print(f"  skip {m}: missing metrics")
            continue
        with open(f32p, encoding="utf-8") as f:
            f32 = json.load(f)
        with open(qp, encoding="utf-8") as f:
            q = json.load(f)

        d = {b: f32["height_bin_ap"][b]["mAP50-95"] - q["height_bin_ap"][b]["mAP50-95"]
             for b in ("XS", "S", "M", "L", "XL")}
        dc = {b: f32["coco_bin_ap"][b]["mAP50-95"] - q["coco_bin_ap"][b]["mAP50-95"]
              for b in ("small", "medium", "large")}

        ap = {b: f32["height_bin_ap"][b]["mAP50-95"] for b in ("XS", "S", "M", "L", "XL")}
        row = {
            "model": m,
            "params_M": count_params(m),
            "ap_fp32_XS": ap["XS"],
            "ap_fp32_S": ap["S"],
            "ap_fp32_XL": ap["XL"],
            "ap_fp32_coco_small": f32["coco_bin_ap"]["small"]["mAP50-95"],
            "map_fp32": f32["overall"]["mAP50-95"],
            "delta_XS": d["XS"], "delta_S": d["S"], "delta_XL": d["XL"],
            "delta_coco_small": dc["small"], "delta_coco_large": dc["large"],
            "SUR_XS_XL": d["XS"] / d["XL"] if abs(d["XL"]) > 1e-6 else float("nan"),
            "SUR_S_XL": d["S"] / d["XL"] if abs(d["XL"]) > 1e-6 else float("nan"),
            "SUR_cocoS_cocoL": dc["small"] / dc["large"] if abs(dc["large"]) > 1e-6 else float("nan"),
            "DIFF_S_XL": d["S"] - d["XL"],
            "DIFF_XS_XL": d["XS"] - d["XL"],
            # Fraction of the bin's OWN FP32 ability that quantization removes.
            # This is the direct answer to the floor-effect objection: if a weak
            # model loses little AP on small signs only because it had little to
            # begin with, its RELATIVE loss should still be large.
            "rel_delta_S": d["S"] / ap["S"] if ap["S"] > 1e-6 else float("nan"),
            "rel_delta_XL": d["XL"] / ap["XL"] if ap["XL"] > 1e-6 else float("nan"),
            "rel_delta_XS": d["XS"] / ap["XS"] if ap["XS"] > 1e-6 else float("nan"),
            "REL_RATIO_S_XL": ((d["S"] / ap["S"]) / (d["XL"] / ap["XL"]))
            if (ap["S"] > 1e-6 and ap["XL"] > 1e-6 and abs(d["XL"] / ap["XL"]) > 1e-6)
            else float("nan"),
        }
        if m in boot:
            for k in ("ci_SUR_XS_XL", "ci_SUR_S_XL", "ci_SUR_cocoS_cocoL"):
                if k in boot[m]:
                    row[k] = boot[m][k]
        rows.append(row)
    return rows


def exact_spearman_p(x, y):
    """Two-sided exact permutation p for Spearman at small n.

    scipy's spearmanr returns an ASYMPTOTIC p-value that is meaningless at n<8
    (it reports p=0 for perfect rank agreement, while the exact two-sided value
    at n=4 is 2/4! = 0.083 and can never reach 0.05). With n<=8 we enumerate all
    n! permutations of y's ranks; that is <=40320 evaluations, trivial.
    """
    from itertools import permutations
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    # A constant predictor/target makes the rank correlation undefined; report
    # that honestly instead of letting spearmanr return nan and the |nan|>=|nan|
    # comparison silently count it as p=0.
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), "undefined_constant"
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    obs = stats.spearmanr(rx, ry).statistic
    if n > 8:
        return float(stats.spearmanr(x, y).pvalue), "asymptotic"
    perms = list(permutations(range(n)))
    count = sum(1 for pm in perms
                if abs(stats.spearmanr(rx, ry[list(pm)]).statistic) >= abs(obs) - 1e-12)
    return count / len(perms), "exact_permutation"


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return None
    sp = stats.spearmanr(x, y)
    pe = stats.pearsonr(x, y)
    p_exact, p_kind = exact_spearman_p(x, y)
    return {"n": int(x.size),
            "spearman_rho": float(sp.statistic), "spearman_p": p_exact, "p_kind": p_kind,
            "pearson_r": float(pe.statistic), "r2": float(pe.statistic ** 2)}


def main(models, precision, metrics_dir, boot_path, out_path):
    rows = load_rows(models, precision, metrics_dir, boot_path)
    if len(rows) < 3:
        raise SystemExit("need >=3 models for any correlation")

    print(f"\n=== Per-model table ({precision} vs FP32, n={len(rows)}) ===")
    print(f"{'model':10}{'params':>8}{'AP_XS':>8}{'AP_S':>8}{'APcocoS':>9}{'mAP':>7}"
          f"{'Δ_XS':>8}{'Δ_S':>8}{'Δ_XL':>8}{'SUR_XS':>8}{'SUR_S':>8}")
    for r in sorted(rows, key=lambda z: z["params_M"]):
        print(f"{r['model']:10}{r['params_M']:8.2f}{r['ap_fp32_XS']:8.3f}{r['ap_fp32_S']:8.3f}"
              f"{r['ap_fp32_coco_small']:9.3f}{r['map_fp32']:7.3f}"
              f"{r['delta_XS']:+8.3f}{r['delta_S']:+8.3f}{r['delta_XL']:+8.3f}"
              f"{r['SUR_XS_XL']:+8.2f}{r['SUR_S_XL']:+8.2f}")

    print(f"\n=== Relative degradation — % of the bin's own FP32 AP that is lost ===")
    print(f"{'model':10}{'params':>8}{'rel Δ_S':>10}{'rel Δ_XL':>10}{'ratio':>8}")
    for r in sorted(rows, key=lambda z: z["params_M"]):
        print(f"{r['model']:10}{r['params_M']:8.2f}{r['rel_delta_S'] * 100:9.1f}%"
              f"{r['rel_delta_XL'] * 100:9.1f}%{r['REL_RATIO_S_XL']:8.2f}")

    # Targets fall into two groups. The RAW per-bin deltas are the clean
    # story: Δ_S and Δ_XL are separate measured quantities, and correlating them
    # against an INDEPENDENT predictor (params) is legitimate. The RATIO targets
    # (SUR) share Δ_XL in their denominator, so any predictor built from Δ_XL is
    # circular — those are quarantined below as diagnostics, never as mechanism.
    targets_clean = {
        "delta_S (abs loss, small)": [r["delta_S"] for r in rows],
        "delta_XL (abs loss, large)": [r["delta_XL"] for r in rows],
        "rel_delta_S (%% of own AP)": [r["rel_delta_S"] for r in rows],
        "rel_delta_XL (%% of own AP)": [r["rel_delta_XL"] for r in rows],
        "DIFF_S_XL": [r["DIFF_S_XL"] for r in rows],
    }
    targets_ratio = {
        "SUR_S_XL": [r["SUR_S_XL"] for r in rows],
        "SUR_cocoS_cocoL": [r["SUR_cocoS_cocoL"] for r in rows],
    }
    # Independent predictor ONLY: parameter count is fixed by architecture and
    # shares no measured term with any delta. Everything AP-based is excluded on
    # purpose — AP_S(FP32) is the baseline inside Δ_S = AP_S(FP32) − AP_S(Q), and
    # even overall mAP(FP32) is baseline-coupled the same way (upward FP32 noise
    # lifts both mAP and every Δ), so regressing a delta on it is change-vs-
    # baseline coupling, not a mechanism.
    predictors = {
        "params_M": [r["params_M"] for r in rows],
        "log10_params": [np.log10(r["params_M"]) for r in rows],
    }

    out = {"precision": precision, "rows": rows, "correlations": {}}
    n = len(rows)
    n_tests = 0
    for group, targets in (("CLEAN (independent predictors)", targets_clean),
                           ("RATIO (interpret with care — see caveats)", targets_ratio)):
        print(f"\n########## {group} ##########")
        for tname, tvals in targets.items():
            print(f"\n=== target: {tname} ===")
            print(f"{'predictor':16}{'spearman':>10}{'p(exact)':>10}{'R²':>8}   trend")
            out["correlations"][tname] = {}
            for pname, pvals in predictors.items():
                c = corr(pvals, tvals)
                if c is None:
                    continue
                out["correlations"][tname][pname] = c
                n_tests += 1
                flag = "strong monotone" if abs(c["spearman_rho"]) >= 0.9 else (
                    "monotone-ish" if abs(c["spearman_rho"]) >= 0.7 else "")
                print(f"{pname:16}{c['spearman_rho']:+10.3f}{c['spearman_p']:10.3f}"
                      f"{c['r2']:8.3f}   {flag}")

    # denominator-sensitivity DIAGNOSTIC, explicitly NOT a mechanism claim
    print(f"\n=== DIAGNOSTIC (not a mechanism): does SUR just track 1/Δ_XL? ===")
    dxl = [r["delta_XL"] for r in rows]
    for tname, tvals in targets_ratio.items():
        c = corr(dxl, tvals)
        if c:
            print(f"  {tname:16} vs Δ_XL (its own denominator): spearman "
                  f"{c['spearman_rho']:+.3f} — circular by construction, shows the "
                  f"ratio is denominator-driven")

    print(f"\nCAVEAT: n={n} models, {n_tests} correlations tested. "
          f"With n<8 these p-values are descriptive only; "
          f"Spearman is the one to read (rank, robust). A perfect |rho|=1 at n=4 "
          f"has p=0.083 and CANNOT reach significance — report as trend, not proof.")

    # Raw facts, printed with the ACTUAL correlations — no hardcoded verdict.
    # The interpretation line is emitted only if the data actually shows the
    # pattern (Δ_XL monotone-decreasing with capacity AND rel Δ_S roughly flat);
    # otherwise it says the pattern does not hold, so adding a non-conforming
    # model can't leave a stale conclusion behind.
    print("\n=== HEADLINE FACTS (raw, uncoupled) ===")
    ss = sorted(rows, key=lambda z: z["params_M"])
    rho_xl = out["correlations"]["delta_XL (abs loss, large)"]["params_M"]["spearman_rho"]
    rho_rel_s = out["correlations"]["rel_delta_S (%% of own AP)"]["params_M"]["spearman_rho"]
    rel_s = [r["rel_delta_S"] for r in rows]
    print(f"  Δ_XL (large-sign loss): {ss[0]['delta_XL']:.3f} @ {ss[0]['params_M']:.1f}M "
          f"-> {ss[-1]['delta_XL']:.3f} @ {ss[-1]['params_M']:.1f}M  (ρ vs params ={rho_xl:+.2f})")
    print(f"  rel Δ_S (small-sign loss, %% of own AP): "
          f"{min(rel_s) * 100:.0f}–{max(rel_s) * 100:.0f}%% across the range (ρ vs params ={rho_rel_s:+.2f})")
    pattern_holds = rho_xl <= -0.8 and abs(rho_rel_s) < 0.6
    if pattern_holds:
        print("  => HOLDS: capacity buys back large-sign robustness while small-sign "
              "loss stays roughly capacity-invariant. SUR only re-expresses this.")
    else:
        print(f"  => PATTERN DOES NOT HOLD at this n (need ρ(Δ_XL)<=-0.8 and "
              f"|ρ(relΔ_S)|<0.6; got {rho_xl:+.2f}, {rho_rel_s:+.2f}). Do not claim it.")
    out["pattern_holds"] = bool(pattern_holds)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precision", default="int8_ptq")
    ap.add_argument("--metrics-dir", default=os.path.join(REPO_ROOT, "metrics"))
    ap.add_argument("--boot", default=os.path.join(REPO_ROOT, "metrics", "sur_bootstrap_int8_ptq.json"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "metrics", "mechanism.json"))
    args = ap.parse_args()
    main(args.models, args.precision, args.metrics_dir, args.boot, args.out)
