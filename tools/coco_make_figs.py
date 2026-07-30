"""Paper figures from canonical 5090 metrics (README_journal v2; dataviz-skill styled).

Fig 1  fig_coco_diff.pdf : DIFF = dAP_S - dAP_L under INT8 vs capacity (log-x),
       three CNN lineages with 95% bootstrap CIs; filled marker = CI excludes 0
       (same convention as the TT100K figure); shaded 20-25M transition band.
Fig 2  fig_coco_fan.pdf  : per-stratum dAP vs capacity, YOLO11 canonical,
       INT8 (left) vs FP8 (right), shared y — FP8 flatness at a glance.

Colors: Okabe-Ito trio (CVD-validated), identity doubled by marker shape.
Usage:  python tools/coco_make_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M5090 = os.path.join(ROOT, "metrics", "coco_5090")
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

FAM = {
    "YOLO11": dict(color="#0072B2", marker="o",
                   models=[("yolo11n", 2.6), ("yolo11s", 9.4), ("yolo11m", 20.1), ("yolo11l", 25.3), ("yolo11x", 56.9)],
                   boot="boot2k_int8_yolo11.json"),
    "YOLOv8": dict(color="#D55E00", marker="s",
                   models=[("yolov8n", 3.2), ("yolov8s", 11.2), ("yolov8m", 25.9), ("yolov8l", 43.7), ("yolov8x", 68.2)],
                   boot="boot2k_int8_v8.json"),
    "YOLO26": dict(color="#009E73", marker="^",
                   models=[("yolo26n", 2.4), ("yolo26s", 9.5), ("yolo26m", 20.4), ("yolo26l", 24.8), ("yolo26x", 55.7)],
                   boot="boot2k_int8_26.json"),
}

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
})


def boot(fname):
    with open(os.path.join(M5090, fname), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ Fig 1
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.axhspan(-0.0005, 0.0005, color="none")
ax.axvspan(20, 25.3, color="0.955", zorder=0)
ax.axhline(0, color="0.35", lw=0.8, ls=(0, (4, 3)), zorder=1)

# The pooled fit, drawn because the segmented test finds no discontinuity:
# what the data support is a smooth trend in log-capacity, and the sign
# change is where that trend passes zero rather than a threshold effect.
_sl = json.load(open(os.path.join(M5090, "capacity_slope.json")))["cnn15_int8"]
_beta = _sl["slope_per_decade"]
_allx = np.array([p for c in FAM.values() for _, p in c["models"]])
_ally = np.array([boot(c["boot"])[m]["DIFF_small_minus_large"]
                  for c in FAM.values() for m, _ in c["models"]])
_a = _ally.mean() - _beta * np.log10(_allx).mean()
_xs = np.logspace(np.log10(_allx.min()), np.log10(_allx.max()), 100)
ax.plot(_xs, _a + _beta * np.log10(_xs), color="0.25", lw=1.1, ls=(0, (6, 2)),
        zorder=1.5, label=f"pooled fit  ${_beta:+.3f}$ AP/decade")

for name, cfg in FAM.items():
    B = boot(cfg["boot"])
    xs, ys, lo, hi, sig = [], [], [], [], []
    for m, p in cfg["models"]:
        r = B[m]
        ci = r["ci_DIFF"]
        xs.append(p); ys.append(r["DIFF_small_minus_large"])
        lo.append(r["DIFF_small_minus_large"] - ci[0]); hi.append(ci[2] - r["DIFF_small_minus_large"])
        sig.append(ci[0] > 0 or ci[2] < 0)
    xs, ys = np.array(xs), np.array(ys)
    ax.plot(xs, ys, color=cfg["color"], lw=1.4, alpha=0.85, zorder=2)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=cfg["color"], elinewidth=0.9,
                capsize=2, capthick=0.9, alpha=0.75, zorder=3)
    for x, y, s in zip(xs, ys, sig):
        ax.plot(x, y, cfg["marker"], ms=6, mew=1.3, mec=cfg["color"],
                mfc=cfg["color"] if s else "white", zorder=4)
    ax.plot([], [], cfg["marker"] + "-", color=cfg["color"], ms=6, mfc=cfg["color"], label=name)

ax.set_xscale("log")
ax.set_xticks([2.5, 5, 10, 20, 40, 70])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Parameters (millions, log scale)")
ax.set_ylabel(r"DIFF $=\Delta AP_{small}-\Delta AP_{large}$  (INT8)")
ax.grid(axis="y", color="0.9", lw=0.6, zorder=0)
ax.legend(loc="upper left", frameon=False, handletextpad=0.5)
ax.annotate("small objects\nhurt more", xy=(0.985, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=7.5, color="0.35")
ax.annotate("large objects\nhurt more", xy=(0.985, 0.03), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=7.5, color="0.35")
ax.annotate("frozen 20M split", xy=(22.4, ax.get_ylim()[0] * 0.93),
            ha="center", fontsize=7, color="0.45")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig_coco_diff.{ext}"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Fig 2
def deltas(model, prec):
    def stats(p):
        with open(os.path.join(M5090, f"{model}_{p}.json"), encoding="utf-8") as f:
            return json.load(f)["stats"]
    r, q = stats("fp32"), stats(prec)
    return [r["AP_small"] - q["AP_small"], r["AP_medium"] - q["AP_medium"], r["AP_large"] - q["AP_large"]]


STRATA = [("small", "#08519C", "-"), ("medium", "#4292C6", "--"), ("large", "#9ECAE1", ":")]
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.7), sharey=True)
for ax, prec, title in zip(axes, ["int8", "fp8"], ["INT8", "FP8 (E4M3)"]):
    xs = [p for _, p in FAM["YOLO11"]["models"]]
    D = np.array([deltas(m, prec) for m, _ in FAM["YOLO11"]["models"]])  # (5,3)
    for j, (name, col, ls) in enumerate(STRATA):
        ax.plot(xs, D[:, j], ls, color=col, lw=1.5, marker="o", ms=4, mfc=col, mec=col)
        if prec == "int8":
            ax.annotate(name, xy=(xs[-1] * 1.08, D[-1, j]), color=col, fontsize=7.5, va="center")
    ax.axhline(0, color="0.35", lw=0.7, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xticks([2.5, 5, 10, 20, 40, 70])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(2.0, 130)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Parameters (M, log)")
    ax.grid(axis="y", color="0.9", lw=0.6, zorder=0)
axes[0].set_ylabel(r"$\Delta AP$ vs FP32 (mAP@[.5:.95])")
axes[1].legend([l for l in axes[1].lines if l.get_linestyle() != (0, (4, 3))][:3], [n for n, _, _ in STRATA], loc="upper right", frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig_coco_fan.{ext}"), dpi=200)
plt.close(fig)

print("-> figs/fig_coco_diff.{pdf,png}, figs/fig_coco_fan.{pdf,png}")
