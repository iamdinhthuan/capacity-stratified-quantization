#!/usr/bin/env python3
"""Compare the FP8 speed advantage across two GPU generations.

Blackwell numbers come from the canonical latency table (5090, TRT 11.1);
Ada numbers from the same nine graphs rebuilt and re-timed here (4090,
TRT 10.16), three repeats in randomised rung order in both cases. The
question is whether "FP8 is the fastest quantized rung" is a property of
the format or of one silicon generation.
"""
import json, os, statistics as st
import numpy as np

R = "/data_nvme/paper"
import sys
ADA = os.path.join(R, sys.argv[1] if len(sys.argv) > 1
                   else "metrics/ada/latency640_ada.jsonl")
TAG = os.path.basename(ADA).replace("latency640_", "").replace(".jsonl", "")
MODELS = ["yolo11n","yolo11m","yolo11x","yolov8n","yolov8m","yolov8x",
          "yolo26n","yolo26m","yolo26x"]
# Blackwell p50 (ms), median of three randomised repeats, as tabulated.
B = {"yolo11n":(0.327,0.411,0.341), "yolo11m":(0.915,0.798,0.757),
     "yolo11x":(1.816,1.793,1.614), "yolov8n":(0.313,0.344,0.300),
     "yolov8m":(0.834,0.736,0.708), "yolov8x":(1.737,1.575,1.546),
     "yolo26n":(0.499,0.516,0.471), "yolo26m":(1.040,0.885,0.834),
     "yolo26x":(1.901,1.812,1.671)}

rows = [json.loads(l) for l in open(ADA)]
def p50(m, p):
    v = [r["latency_p50_ms"] for r in rows
         if r["model"] == m and r["precision"] == p]
    return st.median(v) if v else None

out, ada_fp8_wins, bl_fp8_wins, missing = {}, 0, 0, []
print(f"{'model':9s} | {'Ada p50 (ms)':>26s} | {'Ada vs FP16':>15s} | {'Blackwell vs FP16':>17s}")
print(f"{'':9s} | {'FP16':>8s}{'INT8':>9s}{'FP8':>9s} | {'INT8':>7s}{'FP8':>8s} | {'INT8':>8s}{'FP8':>9s}")
for m in MODELS:
    a = {p: p50(m, p) for p in ("fp16","int8","fp8")}
    if a["fp16"] is None or a["fp8"] is None:
        missing.append(m); continue
    ai8 = a["fp16"]/a["int8"] if a["int8"] else float("nan")
    af8 = a["fp16"]/a["fp8"]
    b16, bi8, bf8 = B[m]
    ada_fp8_wins += a["fp8"] <= min(x for x in a.values() if x)
    bl_fp8_wins  += bf8 <= min(b16, bi8)
    out[m] = {"ada": a, "ada_int8_vs_fp16": ai8, "ada_fp8_vs_fp16": af8,
              "blackwell_int8_vs_fp16": b16/bi8, "blackwell_fp8_vs_fp16": b16/bf8}
    print(f"{m:9s} | {a['fp16']:8.3f}{a['int8'] or float('nan'):9.3f}{a['fp8']:9.3f} | "
          f"{ai8:6.2f}x{af8:7.2f}x | {b16/bi8:7.2f}x{b16/bf8:8.2f}x")
if missing:
    print("\nNO ENGINE (reported as a build failure, not dropped):", missing)
print(f"\nFP8 fastest rung: Ada {ada_fp8_wins}/{len(out)}   Blackwell {bl_fp8_wins}/{len(out)}")
af = np.array([v["ada_fp8_vs_fp16"] for v in out.values()])
bf = np.array([v["blackwell_fp8_vs_fp16"] for v in out.values()])
ai = np.array([v["ada_int8_vs_fp16"] for v in out.values()])
print(f"FP8 vs FP16 speed-up: Ada median {np.median(af):.2f}x (range {af.min():.2f}-{af.max():.2f}), "
      f"Blackwell median {np.median(bf):.2f}x ({bf.min():.2f}-{bf.max():.2f})")
print(f"INT8 vs FP16 on Ada: median {np.median(ai):.2f}x")
print(f"FP8 faster than INT8 on Ada: {sum(1 for v in out.values() if v['ada']['fp8'] < v['ada']['int8'])}/{len(out)}")
json.dump({"per_model": out, "ada_fp8_fastest": ada_fp8_wins,
           "blackwell_fp8_fastest": bl_fp8_wins, "missing_engines": missing,
           "device": "RTX 4090 / TensorRT 10.16 / SM89",
           "protocol": "3 repeats, randomised rung order, 100 warm-up + 500 timed"},
          open(os.path.join(R, f"metrics/ada/{TAG}_vs_blackwell.json"), "w"), indent=1)
print(f"-> metrics/ada/{TAG}_vs_blackwell.json")
