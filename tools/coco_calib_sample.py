"""Build the fixed 512-image COCO calibration set (README_journal.md v2 §3.2).

Mirrors tools/build_calib.py's greedy size-balanced strategy, but reads COCO
instances_train2017.json (native mask-area size bins) instead of YOLO labels,
and — because the pilot box does not need the 19 GB train2017.zip — downloads
just the selected jpgs from images.cocodataset.org. The chosen file list is
committed (calib_list.json) so the 5090 box and the Jetsons reuse the exact
same set later.

Usage:
    python tools/coco_calib_sample.py --n 512 --seed 0
"""
import argparse
import collections
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

from coco_common import COCO_CALIB_DIR, GT_TRAIN

BINS = [("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, float("inf"))]
URL = "http://images.cocodataset.org/train2017/{}"


def bin_of(area):
    for name, lo, hi in BINS:
        if lo <= area < hi:
            return name
    return BINS[-1][0]


def sample_random(n, seed):
    """Uniformly random draw of n annotated train2017 images.

    Used for the calibration-sensitivity ablation. NOTE: the greedy
    size-balanced sample() below is near-deterministic across seeds --- it
    scans `list(remaining)`, whose order comes from set hashing rather than
    from the shuffled list, so two seeds differ by only 1-2 of 512 images.
    An independent draw therefore has to bypass that path entirely.
    """
    print("loading instances_train2017.json (large, ~30s)...")
    with open(GT_TRAIN, encoding="utf-8") as f:
        gt = json.load(f)
    file_of = {im["id"]: im["file_name"] for im in gt["images"]}
    annotated = sorted({a["image_id"] for a in gt["annotations"] if not a.get("iscrowd")})
    rng = random.Random(seed)
    chosen = rng.sample(annotated, n)
    print(f"random draw: {len(chosen)} images (seed {seed})")
    return [file_of[i] for i in chosen]


def sample(n, seed):
    print("loading instances_train2017.json (large, ~30s)...")
    with open(GT_TRAIN, encoding="utf-8") as f:
        gt = json.load(f)
    file_of = {im["id"]: im["file_name"] for im in gt["images"]}
    per_img = collections.defaultdict(collections.Counter)
    for a in gt["annotations"]:
        if a.get("iscrowd"):
            continue
        per_img[a["image_id"]][bin_of(a["area"])] += 1

    rng = random.Random(seed)
    ids = sorted(per_img)
    rng.shuffle(ids)

    # Greedy: keep instance-bin coverage balanced, rarest-filled bin first
    # (same idea as build_calib.py), then top up randomly.
    filled = collections.Counter()
    chosen = []
    remaining = set(ids)
    while len(chosen) < n and remaining:
        rarest = min(BINS, key=lambda b: filled[b[0]])[0]
        best, best_gain = None, -1
        for img_id in list(remaining)[:4000]:  # bounded scan keeps this O(n*4000)
            gain = per_img[img_id][rarest]
            if gain > best_gain:
                best, best_gain = img_id, gain
            if gain >= 8:  # good enough, stop scanning
                break
        if best is None or best_gain <= 0:
            best = remaining.pop() if not best else best
        chosen.append(best)
        remaining.discard(best)
        filled.update(per_img[best])
    print(f"selected {len(chosen)} images; instance coverage per bin: {dict(filled)}")
    return [file_of[i] for i in chosen]


def fetch(files):
    os.makedirs(COCO_CALIB_DIR, exist_ok=True)

    def one(fn):
        dst = os.path.join(COCO_CALIB_DIR, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        r = subprocess.run(["curl", "-sS", "-f", "-o", dst, URL.format(fn)],
                           capture_output=True)
        return r.returncode == 0

    with ThreadPoolExecutor(max_workers=12) as ex:
        ok = list(ex.map(one, files))
    n_ok = sum(ok)
    print(f"downloaded {n_ok}/{len(files)} calibration images -> {COCO_CALIB_DIR}")
    if n_ok < len(files):
        bad = [f for f, o in zip(files, ok) if not o]
        print("FAILED:", bad[:10])
    return n_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategy", choices=["balanced", "random"], default="balanced")
    ap.add_argument("--out-dir", default=None, help="override COCO_CALIB_DIR")
    args = ap.parse_args()
    if args.out_dir:
        COCO_CALIB_DIR = args.out_dir
    files = (sample_random if args.strategy == "random" else sample)(args.n, args.seed)
    out = os.path.join(os.path.dirname(COCO_CALIB_DIR), "calib_list.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": args.n, "seed": args.seed, "files": files}, f, indent=1)
    print(f"-> {out}")
    fetch(files)
