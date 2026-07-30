"""Build a fixed, size-balanced calibration set (README §7.4).

Random sampling over-represents large/common signs (most train images are
dominated by M/L instances), so we greedily prioritize images that cover
under-filled height bins — rarest bin first — before topping up with random
fill. The result is copied to data/calib/images/ and a matching Ultralytics
dataset yaml (configs/tt100k_calib.yaml) is written so every precision that
needs calibration (INT8-PTQ, FP8, FP4) can point at the exact same set.

Usage:
    python tools/build_calib.py --data-root /home/thuan/traffic/data/TT100K \
        --out-dir /home/thuan/traffic/data/calib --n 300
"""
import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict

from common import REPO_ROOT, bin_of, image_size, list_images, load_size_bins, read_yolo_label


def build(data_root, out_dir, n, seed, strategy="balanced"):
    random.seed(seed)
    size_bins = load_size_bins()["height_bins"]
    bin_names = [b["name"] for b in size_bins]

    images_dir = os.path.join(data_root, "images", "train")
    labels_dir = os.path.join(data_root, "labels", "train")
    files = list_images(images_dir)

    img_bins = {}       # path -> set of bin names present
    pop_bin_count = Counter()  # instance count per bin, whole train set
    bin_to_images = defaultdict(list)
    for img_path in files:
        w, h = image_size(img_path)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        bins_here = set()
        for cls_id, xc, yc, bw, bh in read_yolo_label(label_path):
            b = bin_of(bh * h, size_bins)
            pop_bin_count[b] += 1
            bins_here.add(b)
        if bins_here:
            img_bins[img_path] = bins_here
            for b in bins_here:
                bin_to_images[b].append(img_path)

    total_pop = sum(pop_bin_count.values())
    selected = []
    selected_set = set()
    covered_count = Counter()

    if strategy == "random":
        # Neutral baseline: plain uniform sampling over images that have at
        # least one instance, no size-based stratification at all. Used to
        # tell apart "quantization genuinely hurts size X" from "our
        # balanced calib set's bin skew happens to help/hurt size X".
        pool = list(img_bins.keys())
        random.shuffle(pool)
        selected = pool[:n]
        selected_set = set(selected)
        for p in selected:
            for bb in img_bins[p]:
                covered_count[bb] += 1
    else:
        target_per_bin = {b: round(n * (pop_bin_count[b] / total_pop) * 2) for b in bin_names}
        # x2 headroom: greedy pass below stops naturally once `n` images are
        # selected; the inflated target just keeps rare bins from starving early.
        for b in sorted(bin_names, key=lambda x: pop_bin_count[x]):  # rarest bin first
            candidates = [p for p in bin_to_images[b] if p not in selected_set]
            random.shuffle(candidates)
            for p in candidates:
                if covered_count[b] >= target_per_bin[b] or len(selected) >= n:
                    break
                selected.append(p)
                selected_set.add(p)
                for bb in img_bins[p]:
                    covered_count[bb] += 1

        remaining = [p for p in img_bins if p not in selected_set]
        random.shuffle(remaining)
        for p in remaining:
            if len(selected) >= n:
                break
            selected.append(p)
            selected_set.add(p)
            for bb in img_bins[p]:
                covered_count[bb] += 1

    out_images_dir = os.path.join(out_dir, "images")
    # Clear any previous run's images first: Ultralytics calibrates on whatever
    # sits in this directory, not on the manifest, so a leftover larger run would
    # silently pad this calibration set with stale images.
    if os.path.isdir(out_images_dir):
        shutil.rmtree(out_images_dir)
    os.makedirs(out_images_dir, exist_ok=True)
    for p in selected:
        shutil.copy2(p, os.path.join(out_images_dir, os.path.basename(p)))

    manifest = {
        "n_selected": len(selected),
        "n_pool": len(files),
        "population_instance_count_per_bin": dict(pop_bin_count),
        # covered_count increments once per image that contains a bin, so this is
        # images-containing-the-bin, not instance count. Named accordingly.
        "selected_images_containing_bin": dict(covered_count),
        "files": [os.path.basename(p) for p in selected],
    }
    with open(os.path.join(out_dir, "calib_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    yaml_name = f"tt100k_{os.path.basename(os.path.normpath(out_dir))}.yaml"
    yaml_path = os.path.join(REPO_ROOT, "configs", yaml_name)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out_dir}\ntrain: images\nval: images\ntest: images\nnames:\n")
        from common import load_classmap
        for cid, name in load_classmap()["names"].items():
            f.write(f"  {cid}: {name}\n")

    print(f"selected {len(selected)}/{len(files)} images -> {out_images_dir}")
    print("population vs selected instance share per bin:")
    for b in bin_names:
        pop_pct = 100 * pop_bin_count[b] / total_pop
        sel_total = sum(covered_count.values()) or 1
        sel_pct = 100 * covered_count[b] / sel_total
        print(f"  {b:4} pop={pop_pct:5.1f}%  calib={sel_pct:5.1f}%  (n_instances_in_calib={covered_count[b]})")
    print(f"dataset yaml -> {yaml_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategy", choices=["balanced", "random"], default="balanced")
    args = ap.parse_args()
    build(args.data_root, args.out_dir, args.n, args.seed, args.strategy)
