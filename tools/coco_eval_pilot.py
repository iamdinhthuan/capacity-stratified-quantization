"""Standard COCOeval for one (model, precision) prediction file.

Deliberately vanilla — official instances_val2017.json GT, native mask-area
size strata, maxDets=[1,10,100] — so every number is directly comparable to
published COCO AP_S/M/L (README_journal.md v2 §3.2). Writes the 12 stock
COCOeval stats plus the 4 headline fields into metrics/coco_pilot/.

Usage:
    python tools/coco_eval_pilot.py --dt metrics/coco_pilot/pred_yolo11n_int8.json \
        --model yolo11n --precision int8 [--img-ids-from-dt]
"""
import argparse
import json
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from coco_common import GT_VAL, PILOT_METRICS

STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def main(dt_path, model, precision, out_path, img_ids_from_dt):
    coco_gt = COCO(GT_VAL)
    with open(dt_path, encoding="utf-8") as f:
        dets = json.load(f)
    if not dets:
        raise SystemExit(f"{dt_path} is empty")
    coco_dt = coco_gt.loadRes(dets)

    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if img_ids_from_dt:  # smoke tests run on a subset of images
        e.params.imgIds = sorted({d["image_id"] for d in dets})
    e.evaluate()
    e.accumulate()
    e.summarize()

    result = {
        "model": model,
        "precision": precision,
        "n_images": len(e.params.imgIds),
        "stats": {k: float(v) for k, v in zip(STAT_NAMES, e.stats)},
    }
    out_path = out_path or os.path.join(PILOT_METRICS, f"{model}_{precision}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--img-ids-from-dt", action="store_true")
    args = ap.parse_args()
    main(args.dt, args.model, args.precision, args.out, args.img_ids_from_dt)
