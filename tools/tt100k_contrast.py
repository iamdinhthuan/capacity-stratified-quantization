#!/usr/bin/env python3
"""The registered TT100K group contrast, run rather than explained away.

The analysis plan names two co-primary contrasts and controls them with
Holm--Bonferroni. The COCO one was run and reported; the TT100K one was left
as per-model intervals, which leaves the registered multiplicity correction
with nothing to correct. This computes it under the same rule as its COCO
counterpart: one-sided, family-stratified, exact enumeration of every
rearrangement of the low/high labels within each family.

Seven detectors, split at 20M: YOLO11 contributes C(5,2)=10 arrangements and
YOLOv8 contributes C(2,1)=2, so the exact null has 20 points and the smallest
attainable p is 0.05.
"""
import itertools, json, os
import statistics as st

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(R, "metrics")
MODELS = {"yolo11n": (2.7, "yolo11"), "yolo11s": (9.5, "yolo11"),
          "yolo11m": (20.1, "yolo11"), "yolo11l": (25.3, "yolo11"),
          "yolo11x": (57.0, "yolo11"),
          "yolov8s": (11.2, "yolov8"), "yolov8m": (25.9, "yolov8")}
THRESH = 20.0


def diff(m):
    f = json.load(open(os.path.join(M, f"{m}_fp32.json")))
    q = json.load(open(os.path.join(M, f"{m}_int8_ptq.json")))
    h = lambda j, b: j["height_bin_ap"][b]["mAP50-95"]
    return (h(f, "S") - h(q, "S")) - (h(f, "XL") - h(q, "XL"))


def stat(assign):
    """Difference of group means under a labelling: high minus low."""
    hi = [d for d, lab in assign if lab]
    lo = [d for d, lab in assign if not lab]
    return st.mean(hi) - st.mean(lo)


fams = {}
for m, (p, f) in MODELS.items():
    fams.setdefault(f, []).append((diff(m), p >= THRESH, m))

obs = stat([(d, l) for v in fams.values() for d, l, _ in v])

# Exact enumeration: permute the low/high labels within each family, keeping
# each family's group sizes fixed, exactly as the COCO test does.
per_family = []
for f, rows in fams.items():
    n_hi = sum(1 for _, l, _ in rows if l)
    idx = range(len(rows))
    per_family.append([[i in combo for i in idx]
                       for combo in itertools.combinations(idx, n_hi)])

null = []
for choice in itertools.product(*per_family):
    assign = []
    for rows, labels in zip(fams.values(), choice):
        assign += [(rows[i][0], labels[i]) for i in range(len(rows))]
    null.append(stat(assign))

p = sum(1 for v in null if v >= obs) / len(null)
out = {"observed_statistic": obs, "n_arrangements": len(null),
       "p_one_sided_exact": p, "min_attainable_p": 1 / len(null),
       "threshold_M": THRESH,
       "per_model_DIFF": {m: round(diff(m), 5) for m in MODELS},
       "groups": {f: {"high": [m for _, l, m in v if l],
                      "low": [m for _, l, m in v if not l]} for f, v in fams.items()}}
json.dump(out, open(os.path.join(M, "tt100k_contrast.json"), "w"), indent=1)
for m in MODELS:
    print(f"  {m:9s} {MODELS[m][0]:5.1f}M  DIFF={diff(m):+.4f}  "
          f"{'high' if MODELS[m][0] >= THRESH else 'low'}")
print(f"\nobserved high-minus-low = {obs:+.4f}")
print(f"exact one-sided p = {p:.4f} over {len(null)} arrangements "
      f"(floor {1/len(null):.3f})")
print("->", os.path.join(M, "tt100k_contrast.json"))
