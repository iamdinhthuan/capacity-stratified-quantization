"""Calibration-sensitivity ablation with GENUINELY independent draws.

The greedy size-balanced sampler is near-deterministic across seeds (two seeds
differ by 1-2 of 512 images), so the earlier "seed" ablation did not test what
it claimed. This script draws two independent uniform-random 512-image
calibration sets, re-quantizes YOLO11 n/m/x to INT8 with each, and reports the
resulting DIFF against the headline (balanced) calibration set.

Run on the 5090 from ~/coco_journal:
    python tools/calib_indep_run.py
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import onnx

sys.path.insert(0, os.path.expanduser("~/coco_journal/tools"))
os.chdir(os.path.expanduser("~/coco_journal"))

import coco_calib_sample as cs
from coco_common import REPO_ROOT, preprocess
from modelopt.onnx.quantization import quantize as mq

PY = os.path.expanduser("~/miniconda3/envs/qtsd/bin/python")
MODELS = ("yolo11n", "yolo11m", "yolo11x")
SEEDS = (101, 202)


def run(cmd):
    if subprocess.run(cmd, shell=True).returncode != 0:
        print("CMD FAIL:", cmd, flush=True)
        return False
    return True


def overlap(a, b):
    return len(set(a) & set(b))


base_list = json.load(open("data/coco/calib/calib_list.json"))["files"]

for seed in SEEDS:
    d = os.path.join(REPO_ROOT, "data", "coco", f"calib_r{seed}", "images")
    lst = os.path.join(os.path.dirname(d), "calib_list.json")
    if len(glob.glob(os.path.join(d, "*.jpg"))) < 512:
        files = cs.sample_random(512, seed)
        os.makedirs(d, exist_ok=True)
        json.dump({"n": 512, "seed": seed, "strategy": "random", "files": files},
                  open(lst, "w"))
        cs.COCO_CALIB_DIR = d
        cs.fetch(files)
    files = json.load(open(lst))["files"]
    print(f"[seed {seed}] overlap with headline set: {overlap(files, base_list)}/512", flush=True)

    for m in MODELS:
        tag = f"int8_rnd{seed}"
        out = f"exports/coco_pilot/{m}_{tag}.onnx"
        if not os.path.exists(out):
            fs = sorted(glob.glob(os.path.join(d, "*.jpg")))[:512]
            calib = np.stack([preprocess(f, 640)[0][0] for f in fs]).astype(np.float32)
            p = f"exports/coco_pilot/{m}.onnx"
            inp = onnx.load(p, load_external_data=False).graph.input[0].name
            mq(p, quantize_mode="int8", calibration_data={inp: calib},
               calibration_method="max", calibration_eps=["cpu"],
               output_path=out, op_types_to_exclude=["Sigmoid"])
        plan = f"exports/coco_pilot/{m}_{tag}.plan"
        os.path.exists(plan) or run(f"{PY} tools/build_engine.py --onnx {out} --engine {plan}")
        pred = f"metrics/coco_pilot/pred_{m}_{tag}.json"
        os.path.exists(pred) or run(f"{PY} tools/coco_infer_trt.py --engine {plan} --out {pred}")
        ev = f"metrics/coco_pilot/{m}_{tag}.json"
        os.path.exists(ev) or run(f"{PY} tools/coco_eval_pilot.py --dt {pred} --model {m} --precision {tag} > /dev/null")
        print(f"INDEP_DONE {m} {tag}", flush=True)


def diff(m, p):
    r = json.load(open(f"metrics/coco_pilot/{m}_fp32.json"))["stats"]
    q = json.load(open(f"metrics/coco_pilot/{m}_{p}.json"))["stats"]
    return (r["AP_small"] - q["AP_small"]) - (r["AP_large"] - q["AP_large"]), r["AP"] - q["AP"]


print("\n=== calibration sensitivity (independent random draws) ===")
summary = {}
for m in MODELS:
    row = {}
    for p in ("int8",) + tuple(f"int8_rnd{s}" for s in SEEDS):
        try:
            dd, ll = diff(m, p)
            row[p] = {"DIFF": dd, "loss": ll}
            print(f"{m} {p}: loss={ll:+.4f} DIFF={dd:+.4f}")
        except FileNotFoundError:
            print(f"{m} {p}: MISSING")
    summary[m] = row
json.dump(summary, open("metrics/coco_pilot/calib_independent.json", "w"), indent=1)
print("-> metrics/coco_pilot/calib_independent.json")
print("INDEP_ABL_DONE")
