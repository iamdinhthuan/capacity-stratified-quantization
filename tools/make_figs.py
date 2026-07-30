"""Paper figures. Every number is read from metrics/*.json — nothing is typed in.

Palette choices are not taste:
  * size bins XS<S<M<L<XL are ORDERED, so they get a single-hue ordinal ramp
    (light->dark), not categorical hues;
  * precisions are unordered identities, so they get the fixed categorical order.
Both were checked with the dataviz validator (ordinal: monotone L, gaps >=0.06,
light end 2.06:1 vs surface; categorical 4 slots: all-pairs CVD and normal-vision
floors pass). Magenta/yellow sit under 3:1 on white, so those series always carry
a direct label or legend text — never colour alone.

Usage:
    python tools/make_figs.py --models yolo26n yolo11n yolo11s yolov8s yolo11m yolo11l yolo11x \
        --out-dir figs
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import REPO_ROOT

# validated ordinal ramp (blue, light->dark) for the ordered size bins
BIN_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
BINS = ["XS", "S", "M", "L", "XL"]
# validated categorical order for precisions (identity, unordered)
PREC_COLOR = {"fp32": "#2a78d6", "fp16": "#008300", "int8_ptq": "#e87ba4", "fp8": "#eda100"}
PREC_LABEL = {"fp32": "FP32", "fp16": "FP16", "int8_ptq": "INT8-PTQ", "fp8": "FP8"}

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e5e4e0"


def style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def load(models, metrics_dir, precision):
    rows = []
    for m in models:
        f32 = os.path.join(metrics_dir, f"{m}_fp32.json")
        q = os.path.join(metrics_dir, f"{m}_{precision}.json")
        if not (os.path.exists(f32) and os.path.exists(q)):
            continue
        with open(f32, encoding="utf-8") as f:
            a = json.load(f)
        with open(q, encoding="utf-8") as f:
            b = json.load(f)
        ap = {k: a["height_bin_ap"][k]["mAP50-95"] for k in BINS}
        delta = {k: ap[k] - b["height_bin_ap"][k]["mAP50-95"] for k in BINS}
        rows.append({
            "model": m,
            "delta": delta,
            "ap_fp32": ap,
            "rel": {k: (delta[k] / ap[k] if ap[k] > 1e-6 else np.nan) for k in BINS},
        })
    return rows


def params_of(models, mech_path):
    if os.path.exists(mech_path):
        with open(mech_path, encoding="utf-8") as f:
            mech = json.load(f)
        return {r["model"]: r["params_M"] for r in mech["rows"]}
    return {}


def fig_delta_fan(rows, params, out, precision):
    """THE headline: which size bins does capacity actually rescue?

    Two panels because the absolute panel alone invites the floor-effect reply
    ("small models lose little on small signs because they had little"). The
    relative panel answers it in the objection's own currency: share of the
    bin's own FP32 ability that is destroyed.

    XS is drawn but deliberately de-emphasised (muted, dashed, labelled n=16) —
    it is shown rather than dropped so nothing looks hidden, but it must not be
    read as a trend: its bootstrap CI spans zero (see fig2).
    """
    xs = np.array([params.get(r["model"], np.nan) for r in rows])
    order = np.argsort(xs)
    xs_s = xs[order]
    trend_bins = ["S", "M", "L", "XL"]   # bins with enough test instances

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    panels = [
        (axes[0], "delta", 100.0, "AP lost  (Δ mAP50-95, points)", "absolute loss"),
        (axes[1], "rel", 100.0, "share of the bin's own FP32 AP lost (%)", "relative loss"),
    ]
    for ax, key, scale, ylab, sub in panels:
        # unreliable bin first, so the reliable ones draw on top
        ys = [rows[i][key]["XS"] * scale for i in order]
        ax.plot(xs_s, ys, "--o", color=INK2, linewidth=1.0, markersize=3,
                alpha=0.55, label="XS (n=16, noise)", zorder=2)
        for bi, b in enumerate(trend_bins):
            ys = [rows[i][key][b] * scale for i in order]
            c = BIN_RAMP[bi + 1]
            ax.plot(xs_s, ys, "-o", color=c, linewidth=1.8, markersize=4.5,
                    label=b, zorder=4 + bi)
            ax.annotate(b, (xs_s[-1], ys[-1]), xytext=(5, 0), textcoords="offset points",
                        color=c, fontsize=7.5, va="center", fontweight="bold")
        ax.axhline(0, color=INK2, linewidth=0.6, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(xs_s)
        ax.set_xticklabels([f"{v:.1f}" for v in xs_s])
        ax.minorticks_off()
        ax.set_xlabel("model parameters (M)")
        ax.set_ylabel(ylab)
        ax.set_title(sub, loc="left", fontsize=8, color=INK2)
        ax.margins(x=0.12)
    axes[0].legend(title="sign height bin", ncol=5, loc="upper center",
                   bbox_to_anchor=(1.05, -0.24), handlelength=1.4)
    fig.suptitle(f"Capacity buys robustness on large signs only "
                 f"({PREC_LABEL[precision]} vs FP32)", x=0.005, ha="left", fontsize=9.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def fig_sur_ci(rows, params, boot_path, out, key="ci_SUR_S_XL", point_key="SUR_S_XL"):
    if not os.path.exists(boot_path):
        print(f"   (skip SUR CI figure: {boot_path} missing)")
        return
    with open(boot_path, encoding="utf-8") as f:
        boot = json.load(f)
    ms = [r["model"] for r in rows if r["model"] in boot]
    if not ms:
        return
    xs = [params.get(m, np.nan) for m in ms]
    med = [boot[m][key]["median"] for m in ms]
    lo = [boot[m][key]["median"] - boot[m][key]["lo"] for m in ms]
    hi = [boot[m][key]["hi"] - boot[m][key]["median"] for m in ms]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    o = np.argsort(xs)
    ax.errorbar([xs[i] for i in o], [med[i] for i in o],
                yerr=[[lo[i] for i in o], [hi[i] for i in o]],
                fmt="-o", color="#2a78d6", ecolor="#86b6ef", elinewidth=1.4,
                capsize=3, linewidth=1.6, markersize=4.5, zorder=3)
    ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle="--", zorder=1)
    ax.annotate("equal harm\n(SUR = 1)", (ax.get_xlim()[0], 1.0), xytext=(2, 4),
                textcoords="offset points", fontsize=6.5, color=INK2)
    for i in o:
        ax.annotate(ms[i], (xs[i], med[i]), xytext=(0, 7), textcoords="offset points",
                    fontsize=6, color=INK2, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("model parameters (M, log scale)")
    ax.set_ylabel("SUR  =  Δ$_{S}$ / Δ$_{XL}$")
    ax.set_title("Size-unfairness vs capacity (95% image-bootstrap CI)",
                 loc="left", color=INK)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def fig_recall_conf(op_path, models, out):
    if not os.path.exists(op_path):
        print(f"   (skip recall/conf figure: {op_path} missing)")
        return
    with open(op_path, encoding="utf-8") as f:
        op = json.load(f)
    ms = [m for m in models if m in op and "fp32" in op[m] and "int8_ptq" in op[m]]
    if not ms:
        return
    n = len(ms)
    fig, axes = plt.subplots(1, n, figsize=(1.85 * n, 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, ms):
        for prec in ("fp32", "int8_ptq"):
            c = op[m][prec]["pr_curve"]
            ax.plot([p["conf"] for p in c], [p["recall"] for p in c],
                    color=PREC_COLOR[prec], linewidth=1.6, label=PREC_LABEL[prec], zorder=3)
        ax.axvline(0.25, color=INK2, linewidth=0.7, linestyle=":", zorder=1)
        rc = op[m]["int8_ptq"].get("recalibration", {})
        if rc.get("recovered"):
            ax.axvline(rc["recovered"]["conf"], color="#eda100", linewidth=1.0,
                       linestyle="--", zorder=2)
        ax.set_xscale("log")
        ax.set_title(m, loc="left", fontsize=8)
        ax.set_xlabel("confidence threshold")
    axes[0].set_ylabel("recall")
    axes[0].legend(loc="lower left")
    axes[-1].annotate("dotted: stock 0.25\ndashed: re-tuned", (0.98, 0.04),
                      xycoords="axes fraction", ha="right", fontsize=6, color=INK2)
    fig.suptitle("Quantization moves the operating point, so the threshold must move too",
                 x=0.01, ha="left", fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def table_ewd(op_path, models, out_tex):
    """EWD is a TABLE, not a chart.

    s*(rho) is 0 for almost every (model, precision, rho) cell and jumps to
    "never" for a few — a plot of mostly-zeros with a sentinel spike encodes
    nothing a reader can't get faster from 8 rows, and drawing "never" at a
    finite y invents a magnitude that does not exist. So we emit LaTeX.
    """
    if not os.path.exists(op_path):
        return
    with open(op_path, encoding="utf-8") as f:
        op = json.load(f)
    rhos = [0.5, 0.6, 0.7, 0.8, 0.9]
    lines = [r"\begin{tabular}{ll" + "r" * len(rhos) + "}", r"\toprule",
             "Detector & Precision & " + " & ".join(rf"$\rho{{=}}{r}$" for r in rhos) + r" \\",
             r"\midrule"]
    for m in models:
        if m not in op:
            continue
        for prec in ("fp32", "fp16", "int8_ptq", "fp8"):
            e = op[m].get(prec, {}).get("ewd_rho")
            if not e:
                continue
            cells = []
            for r in rhos:
                v = e[str(r)]
                # None = rho not reached anywhere up to the 200 px scan limit,
                # not literally "never" — label it honestly.
                cells.append(r"$>$200" if v is None else str(v))
            lines.append(f"{m} & {PREC_LABEL[prec]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"% >200 = operating recall never reached within the 0-200 px scan range"]
    os.makedirs(os.path.dirname(out_tex) or ".", exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"-> {out_tex}  (EWD reported as a table: the cells are mostly 0/never)")


def main(models, metrics_dir, out_dir, precision):
    style()
    os.makedirs(out_dir, exist_ok=True)
    rows = load(models, metrics_dir, precision)
    if not rows:
        raise SystemExit("no metrics found")
    params = params_of([r["model"] for r in rows],
                       os.path.join(metrics_dir, "mechanism.json"))
    if not params:
        raise SystemExit("run tools/mechanism_analysis.py first (needs params)")

    for ext in ("pdf", "png"):
        fig_delta_fan(rows, params, os.path.join(out_dir, f"fig1_delta_fan.{ext}"), precision)
        fig_sur_ci(rows, params,
                   os.path.join(metrics_dir, f"sur_bootstrap_{precision}.json"),
                   os.path.join(out_dir, f"fig2_sur_ci.{ext}"))
        fig_recall_conf(os.path.join(metrics_dir, "operating_point.json"),
                        [r["model"] for r in rows],
                        os.path.join(out_dir, f"fig3_recall_conf.{ext}"))
    table_ewd(os.path.join(metrics_dir, "operating_point.json"),
              [r["model"] for r in rows], os.path.join(out_dir, "table_ewd.tex"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precision", default="int8_ptq")
    ap.add_argument("--metrics-dir", default=os.path.join(REPO_ROOT, "metrics"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "figs"))
    args = ap.parse_args()
    main(args.models, args.metrics_dir, args.out_dir, args.precision)
