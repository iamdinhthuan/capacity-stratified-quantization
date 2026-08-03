"""Figure 3: FP8 against INT8, and against INT8 given the better calibrator.

One dumbbell per checkpoint. The point of the figure is that the FP8 advantage
is not an artefact of the shipping `max` calibrator: even after `entropy`
recovers half of INT8's loss, FP8 is still the lower-loss rung on every
convolutional model. Left panel is the fifteen convolutional checkpoints, right
panel the seven detection transformers, on their own scale because INT8 there
does not degrade the model but deletes it.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

M = os.path.join(os.path.dirname(__file__), "..", "metrics", "coco_5090")
R = os.path.join(M, "..", "rtdetr")
FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")

def ap(model, prec, root=M):
    f = os.path.join(root, f"{model}_{prec}.json")
    return json.load(open(f))["stats"]["AP"] if os.path.exists(f) else None

CNN = [("yolo11n",2.6),("yolo11s",9.4),("yolo11m",20.1),("yolo11l",25.3),("yolo11x",56.9),
       ("yolov8n",3.2),("yolov8s",11.2),("yolov8m",25.9),("yolov8l",43.7),("yolov8x",68.2),
       ("yolo26n",2.4),("yolo26s",9.5),("yolo26m",20.4),("yolo26l",24.8),("yolo26x",55.7)]
TRF = [("dfine_n",4),("dfine_s",10),("dfine_m",19),("dfine_l",31),("dfine_x",62),
       ("rtdetr_l",32),("rtdetr_x",67)]

def pretty(m):
    return (m.replace("yolo11","YOLO11").replace("yolov8","YOLOv8")
             .replace("yolo26","YOLO26").replace("dfine_","D-FINE-")
             .replace("rtdetr_","RT-DETR-"))

# colour and marker both carry the series, so the figure survives grayscale
SER = [("FP8 (E4M3)",      "fp8",     "#1f77b4", "o"),
       ("INT8, entropy",   "int8ent", "#ff7f0e", "s"),
       ("INT8, max",       "int8",    "#2ca02c", "^")]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0),
                         gridspec_kw={"width_ratios": [1, 1]})

for ax, items, root, title in (
        (axes[0], CNN, M, "Convolutional (15)"),
        (axes[1], TRF, None, "Detection transformers (7)")):
    items = sorted(items, key=lambda t: t[1])
    ys = range(len(items))
    for y, (m, _) in zip(ys, items):
        rt = R if m.startswith("rtdetr") else M
        base = ap(m, "fp32", rt)
        vals = [(lab, (base - ap(m, p, rt)) * 100 if ap(m, p, rt) is not None else None, c, mk)
                for lab, p, c, mk in SER]
        got = [v for v in vals if v[1] is not None]
        if len(got) > 1:
            ax.plot([min(v[1] for v in got), max(v[1] for v in got)], [y, y],
                    color="0.75", lw=1.1, zorder=1)
        for lab, v, c, mk in got:
            ax.scatter(v, y, s=26, color=c, marker=mk, zorder=3,
                       edgecolor="white", linewidth=0.6)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([pretty(m) for m, _ in items], fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", color="0.9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

axes[0].set_xlim(-0.3, 10)
axes[1].set_xlim(-1.5, 56)
h = [plt.Line2D([], [], color=c, marker=mk, ls="", markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.6, label=lab)
     for lab, _, c, mk in SER]
axes[0].legend(handles=h, fontsize=7.5, loc="lower right", frameon=False)
axes[1].annotate("RT-DETR has no entropy arm", xy=(2, 4.55), fontsize=6.5, color="0.45")
fig.tight_layout(rect=[0, 0.055, 1, 1])
fig.supxlabel("mAP points lost against the model's own FP32 engine", fontsize=8.5, y=0.022)
os.makedirs(FIGS, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig_format.{ext}"), dpi=200)
print("-> figs/fig_format.{pdf,png}")
