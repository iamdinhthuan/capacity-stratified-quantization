"""Print the precision-ladder summary for given models from metrics/coco_pilot.

Usage: python tools/coco_ladder_summary.py yolo11n yolo11s ...
"""
import json
import os
import sys

MET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics", "coco_pilot")


def load(m, p):
    with open(os.path.join(MET, f"{m}_{p}.json"), encoding="utf-8") as f:
        return json.load(f)["stats"]


for m in sys.argv[1:]:
    try:
        r = load(m, "fp32")
    except FileNotFoundError:
        print(f"{m}: fp32 pending")
        continue
    for p in ["fp16", "int8", "fp8"]:
        try:
            s = load(m, p)
        except FileNotFoundError:
            continue
        dS = r["AP_small"] - s["AP_small"]
        dL = r["AP_large"] - s["AP_large"]
        print(f"{m:9} {p:5} AP={s['AP']:.4f} loss={r['AP'] - s['AP']:+.4f} "
              f"dS={dS:+.4f} dL={dL:+.4f} DIFF={dS - dL:+.4f}")
