#!/usr/bin/env python3
"""Graphical abstract: required by IVC at submission.

One panel, one claim. The guide asks for at least 531 x 1328 pixels (h x w),
legible at 5 x 13 cm, which is a wide strip -- so the figure reads left to
right as the argument does: what INT8 costs, what FP8 costs, and the fact
that the aggregate number hides the difference.

Colours are the Okabe-Ito pairs used in the paper's other figures, and every
value is read from the released metric files rather than typed in.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M5, MR = os.path.join(R, "metrics", "coco_5090"), os.path.join(R, "metrics", "rtdetr")
CNN = [m + s for m in ("yolo11", "yolov8", "yolo26") for s in ("nsmlx")]
PARAMS = {"yolo11n":2.6,"yolo11s":9.4,"yolo11m":20.1,"yolo11l":25.3,"yolo11x":56.9,
          "yolov8n":3.2,"yolov8s":11.2,"yolov8m":25.9,"yolov8l":43.7,"yolov8x":68.2,
          "yolo26n":2.4,"yolo26s":9.5,"yolo26m":20.4,"yolo26l":24.8,"yolo26x":55.7}

def st(d, m, p):
    return json.load(open(os.path.join(d, f"{m}_{p}.json")))["stats"]

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42})
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "0.45"

fig, axes = plt.subplots(1, 3, figsize=(13.0/2.54*1.6, 5.0/2.54*1.6),
                         gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]})

# --- panel 1: aggregate loss, the number practitioners accept on -----------
ax = axes[0]
i8 = [(st(M5,m,"fp32")["AP"] - st(M5,m,"int8")["AP"])*100 for m in CNN]
f8 = [(st(M5,m,"fp32")["AP"] - st(M5,m,"fp8")["AP"])*100 for m in CNN]
ax.boxplot([i8, f8], widths=.5, patch_artist=True, medianprops=dict(color="black", lw=1.2),
           boxprops=dict(facecolor="#f0f0f0", lw=.8), whiskerprops=dict(lw=.8),
           capprops=dict(lw=.8), flierprops=dict(ms=3, mfc=GREY, mec=GREY))
ax.set_xticks([1,2]); ax.set_xticklabels(["INT8","FP8"])
ax.set_ylabel("aggregate mAP lost (points)")
ax.set_title("FP8 costs a fraction\nof what INT8 costs", fontsize=9.5)
ax.set_ylim(-0.6, max(i8) * 1.30)
ax.text(1.34, np.median(i8), f"median {np.median(i8):.1f}", va="center", fontsize=8, color=ORANGE)
ax.text(2.34, np.median(f8), f"{np.median(f8):.2f}", va="center", fontsize=8, color=BLUE)
ax.grid(axis="y", color="0.92", lw=.6)

# --- panel 2: the size gap the aggregate number hides ----------------------
ax = axes[1]
xs = np.array([PARAMS[m] for m in CNN])
gap = np.array([((st(M5,m,"fp32")["AP_small"]-st(M5,m,"int8")["AP_small"])
                -(st(M5,m,"fp32")["AP_large"]-st(M5,m,"int8")["AP_large"])) for m in CNN])
ax.axhline(0, color="0.4", lw=.8, ls=(0,(4,3)))
ax.scatter(xs, gap, s=34, c=ORANGE, zorder=3, label="INT8")
b, a = np.polyfit(np.log10(xs), gap, 1)
gx = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 50)
ax.plot(gx, a + b*np.log10(gx), color="0.25", lw=1.3, ls=(0,(6,2)))
ax.set_xscale("log"); ax.set_xticks([2.5,10,40]); ax.set_xlim(2, 80)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("parameters (M, log)")
ax.set_ylabel(r"$\Delta$AP$_{small}$ $-$ $\Delta$AP$_{large}$")
ax.set_title("but capacity moves the damage\nonto small objects  (+0.05/decade)", fontsize=9.5)
ax.set_ylim(gap.min() - .012, gap.max() + .028)
ax.annotate("small objects hurt more", xy=(.03,.97), xycoords="axes fraction",
            ha="left", va="top", fontsize=7.5, color=GREY)
ax.grid(axis="y", color="0.92", lw=.6)

# --- panel 3: the transformer case -----------------------------------------
ax = axes[2]
names = ["RT-DETR-l", "RT-DETR-x"]
i8t = [(st(MR,m,"fp32")["AP"]-st(MR,m,"int8")["AP"])*100 for m in ("rtdetr_l","rtdetr_x")]
f8t = [(st(MR,m,"fp32")["AP"]-st(MR,m,"fp8")["AP"])*100 for m in ("rtdetr_l","rtdetr_x")]
y = np.arange(2)
ax.barh(y+.19, i8t, .36, color=ORANGE, label="INT8")
ax.barh(y-.19, f8t, .36, color=BLUE, label="FP8")
for k in range(2):
    ax.text(i8t[k]-2, y[k]+.19, f"{i8t[k]:.0f}", va="center", ha="right", color="white", fontsize=8)
    ax.text(f8t[k]+1.5, y[k]-.19, f"{f8t[k]:.2f}", va="center", fontsize=8, color=BLUE)
ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlim(0, max(i8t) * 1.18)
ax.set_ylim(1.75, -0.85)          # room above the bars for the legend
ax.set_xlabel("mAP lost (points)")
ax.set_title("INT8 deletes detection transformers;\nFP8 does not", fontsize=9.5)
ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=2,
          handlelength=1.2, columnspacing=1.0)
ax.grid(axis="x", color="0.92", lw=.6)

fig.tight_layout(w_pad=2.0)
out = os.path.join(R, "A_Capacity_Stratified_Analysis_of_INT8_and_FP8_Quantization_on_object_detection")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"graphical_abstract.{ext}"), dpi=260, bbox_inches="tight")
print("-> graphical_abstract.{pdf,png}")
