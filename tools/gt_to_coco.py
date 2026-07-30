"""Build a COCO-format ground-truth JSON from TT100K YOLO labels (test split).

image_id = index in the sorted file list, kept stable so tools/infer.py can
reuse the same ids when writing predictions. Each annotation carries extra
(non-standard, but harmless to pycocotools) fields: height_px, superclass, in45.

Usage:
    python tools/gt_to_coco.py --data-root /home/thuan/traffic/data/TT100K \
        --split test --out metrics/gt_test.json
"""
import argparse
import json
import os

from common import (
    REPO_ROOT,
    image_size,
    list_images,
    load_classmap,
    read_yolo_label,
    superclass_of,
    yolo_to_xywh_abs,
)


def build(data_root, split, out_path):
    classmap = load_classmap()
    names = classmap["names"]
    classes_45 = {c["id"] for c in classmap["classes_45"]}

    images_dir = os.path.join(data_root, "images", split)
    labels_dir = os.path.join(data_root, "labels", split)
    files = list_images(images_dir)
    if not files:
        raise SystemExit(f"No images found in {images_dir}")

    categories = [
        {"id": cid, "name": name, "superclass": superclass_of(name, classmap), "in45": cid in classes_45}
        for cid, name in names.items()
    ]

    images, annotations = [], []
    # Annotation ids MUST start at 1: pycocotools stores the matched GT id in
    # dtMatches and later treats it as a boolean TP flag, so a detection that
    # matches annotation id 0 is silently scored as a false positive. (Verified:
    # shifting ids 0->1 moved overall mAP by 3e-6, i.e. exactly one instance.)
    ann_id = 1
    for img_id, img_path in enumerate(files):
        w, h = image_size(img_path)
        images.append({"id": img_id, "file_name": os.path.basename(img_path), "width": w, "height": h})

        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        for cls_id, xc, yc, bw, bh in read_yolo_label(label_path):
            bbox = yolo_to_xywh_abs(xc, yc, bw, bh, w, h)
            # Clip to image bounds. TT100K has 29 GT boxes spilling up to ~13 px
            # outside the frame; Ultralytics clips its detections, so an un-clipped
            # GT could be unmatchable at high IoU (and its inflated area could even
            # land it in the wrong size bin). Clip so GT and detections share a frame.
            x0, y0 = max(0.0, bbox[0]), max(0.0, bbox[1])
            x1 = min(float(w), bbox[0] + bbox[2])
            y1 = min(float(h), bbox[1] + bbox[3])
            bbox = [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]
            height_px = bbox[3]
            name = names[cls_id]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "height_px": height_px,
                "superclass": superclass_of(name, classmap),
                "in45": cls_id in classes_45,
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    n_with_obj = len({a["image_id"] for a in annotations})
    print(f"{split}: {len(images)} images, {len(annotations)} instances, "
          f"{n_with_obj} images with >=1 instance -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "metrics", "gt_test.json"))
    args = ap.parse_args()
    build(args.data_root, args.split, args.out)
