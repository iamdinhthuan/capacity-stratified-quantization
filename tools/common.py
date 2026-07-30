"""Shared helpers for the stratified-eval harness (README §8)."""
import glob
import os
import yaml
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_classmap(path=None):
    path = path or os.path.join(REPO_ROOT, "configs", "classmap.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_size_bins(path=None):
    path = path or os.path.join(REPO_ROOT, "configs", "size_bins.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def superclass_of(name, classmap):
    prefix_map = classmap["superclass_prefix_map"]
    for prefix, sc in prefix_map.items():
        if name.startswith(prefix):
            return sc
    return classmap["default_superclass"]


def bin_of(value, bins):
    for b in bins:
        lo, hi = b["min"], b["max"]
        if lo <= value < hi:
            return b["name"]
    return bins[-1]["name"]


def list_images(images_dir):
    files = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )
    return files


def image_size(path):
    with Image.open(path) as im:
        return im.size  # (w, h)


def read_yolo_label(label_path):
    """Yields (cls_id, xc, yc, w, h) normalized, or nothing if file absent/empty."""
    if not os.path.exists(label_path):
        return
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = (float(x) for x in parts[1:5])
            yield cls_id, xc, yc, w, h


def yolo_to_xywh_abs(xc, yc, w, h, img_w, img_h):
    """Normalized YOLO box -> absolute COCO-style [x, y, w, h]."""
    bw, bh = w * img_w, h * img_h
    x = xc * img_w - bw / 2
    y = yc * img_h - bh / 2
    return [x, y, bw, bh]
