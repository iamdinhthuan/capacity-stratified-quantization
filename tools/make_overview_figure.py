"""Methodology overview: checkpoint -> precision ladder -> engines -> validity
checks -> evaluation. One horizontal strip, four blocks. The two data inputs
are drawn separately on purpose: the calibration subset touches quantization
only, the evaluation set touches evaluation only, so the figure itself rules
out the leakage a reviewer would otherwise have to check for."""
import os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")
GREY, EDGE, CHIP = "#f4f4f4", "0.25", "#dce8f2"

fig, ax = plt.subplots(figsize=(7.5, 2.9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def block(x0, x1, y0, y1, title, lines, fill=GREY, ls="-", lw=1.0, tfs=7.3, lfs=6.6, gap=0.075, ts=0.155):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
        boxstyle="round,pad=0.004", fc=fill, ec=EDGE, ls=ls, lw=lw, zorder=2))
    ax.text((x0+x1)/2, y1-0.045, title, ha="center", va="top",
            fontsize=tfs, fontweight="bold", zorder=3)
    for i, ln in enumerate(lines):
        ax.text((x0+x1)/2, y1-ts-gap*i, ln, ha="center", va="top",
                fontsize=lfs, zorder=3)

def arrow(x0, y0, x1, y1, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=9, color=EDGE, lw=1.0, zorder=4, **kw))

Y0, Y1 = 0.33, 0.96
# -- checkpoint input ----------------------------------------------------
block(0.010, 0.115, 0.53, 0.79, "", [], fill="white")
ax.text(0.0625, 0.660, "official /\ntrained\ncheckpoint", ha="center", va="center", fontsize=6.4)
arrow(0.115, 0.66, 0.148, 0.66)
# -- A: export & quantization -------------------------------------------
block(0.148, 0.360, Y0, Y1, "1. Export and quantization",
      ["ONNX export, one graph", "", "", "", "Q/DQ graphs,", "quantized once"])
for i, p in enumerate(["FP32", "FP16", "INT8", "FP8"]):
    x = 0.160 + i*0.0495
    ax.add_patch(FancyBboxPatch((x, 0.565), 0.043, 0.115,
        boxstyle="round,pad=0.003", fc=CHIP, ec="black", lw=1.5, zorder=5))
    ax.text(x+0.0215, 0.622, p, ha="center", va="center", fontsize=5.9, zorder=6)
arrow(0.360, 0.65, 0.395, 0.65)
# -- B: engines ----------------------------------------------------------
block(0.395, 0.585, Y0, Y1, "2. TensorRT engines",
      ["one engine per rung,", "built per device,", "strongly typed", "", "shared preprocessing,", "decoding, post-processing"])
arrow(0.585, 0.65, 0.620, 0.65)
# -- C: validity checks --------------------------------------------------
block(0.620, 0.800, Y0, Y1, "3. Deployment validity",
      ["FP16 consistency", "with FP32", "FP32 engine fidelity", "vs. reference", "catastrophic-collapse", "screen"])
arrow(0.800, 0.65, 0.835, 0.65)
# -- D: evaluation -------------------------------------------------------
block(0.835, 0.995, Y0, Y1, "4. Evaluation",
      ["overall AP;", "AP$_S$ / AP$_M$ / AP$_L$", "$\\Delta$AP, DIFF, paired", "bootstrap intervals", "latency, power,", "energy per image"])
# -- data inputs ---------------------------------------------------------
block(0.148, 0.360, 0.020, 0.24, "calibration subset",
      ["fixed image list;", "INT8 and FP8 ranges only"], fill="white", ls=(0,(3,2)), tfs=6.6, lfs=5.9, gap=0.062, ts=0.105)
arrow(0.254, 0.24, 0.254, 0.32)
block(0.620, 0.995, 0.020, 0.24, "evaluation set",
      ["COCO val2017 / TT100K test;", "never used for calibration"], fill="white", ls=(0,(3,2)), tfs=6.6, lfs=5.9, gap=0.062, ts=0.105)
arrow(0.885, 0.24, 0.885, 0.32)
# -- legend ---------------------------------------------------------------
ax.add_patch(FancyBboxPatch((0.408, 0.150), 0.018, 0.062, boxstyle="round,pad=0.003",
             fc=CHIP, ec="black", lw=1.5, zorder=5))
ax.text(0.433, 0.181, "experimental factor (precision)", fontsize=5.9, va="center")
ax.add_patch(FancyBboxPatch((0.408, 0.055), 0.018, 0.062, boxstyle="round,pad=0.003",
             fc=GREY, ec=EDGE, lw=1.0, zorder=5))
ax.text(0.433, 0.086, "controlled, shared across rungs", fontsize=5.9, va="center")
fig.tight_layout(pad=0.3)
os.makedirs(FIGS, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig_overview.{ext}"), dpi=220, bbox_inches="tight")
print("-> figs/fig_overview.{pdf,png}")
