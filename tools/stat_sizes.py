"""Histogram of instance counts per (height-bin x super-class), per split.
Run this FIRST (README §5.3) to know if bins have enough samples before
committing to the stratified-eval design.

Usage:
    python tools/stat_sizes.py --data-root /home/thuan/traffic/data/TT100K --split test
"""
import argparse
import os
from collections import Counter

from common import bin_of, image_size, list_images, load_classmap, load_size_bins, read_yolo_label, superclass_of


def run(data_root, split):
    classmap = load_classmap()
    names = classmap["names"]
    size_bins = load_size_bins()["height_bins"]

    images_dir = os.path.join(data_root, "images", split)
    labels_dir = os.path.join(data_root, "labels", split)
    files = list_images(images_dir)

    bin_sc_count = Counter()
    bin_count = Counter()
    n_instances = 0
    for img_path in files:
        w, h = image_size(img_path)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        for cls_id, xc, yc, bw, bh in read_yolo_label(label_path):
            height_px = bh * h
            b = bin_of(height_px, size_bins)
            sc = superclass_of(names[cls_id], classmap)
            bin_sc_count[(b, sc)] += 1
            bin_count[b] += 1
            n_instances += 1

    print(f"split={split}  images={len(files)}  instances={n_instances}\n")
    print(f"{'bin':6}{'count':>10}   per super-class")
    for b in size_bins:
        name = b["name"]
        scs = ", ".join(f"{sc}={bin_sc_count[(name, sc)]}" for sc in sorted({s for (_, s) in bin_sc_count}))
        print(f"{name:6}{bin_count[name]:>10}   {scs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    run(args.data_root, args.split)
