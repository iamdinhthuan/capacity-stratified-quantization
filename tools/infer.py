"""Run a trained detector over a test split and dump COCO-format predictions.

Works for any weights Ultralytics' YOLO() can load: .pt checkpoints, and
Ultralytics-exported .engine files (FP16 / INT8-PTQ / INT8-QAT). Raw
trtexec-built FP8/FP4 .plan files (README §7.3) are NOT loadable this way —
those will need a separate TensorRT-Python-API loader in a later script.

Default batch=1: Ultralytics exports TensorRT engines with a static batch
size of 1 unless dynamic=True is passed at export time, so a batch>1 chunk
crashes with a shape-mismatch assert against .pt-only defaults. Accuracy
eval doesn't need throughput, so bs=1 keeps this correct across every
precision without touching the export config; tools/measure_system.py
covers actual bs=1 throughput/latency separately.

image_id matches tools/gt_to_coco.py (index in the sorted file list of the
same images_dir), so predictions line up with the GT file without extra bookkeeping.

Usage:
    python tools/infer.py --model runs/detect/runs/yolo26n_fp32/weights/best.pt \
        --images-dir /home/thuan/traffic/data/TT100K/images/test \
        --imgsz 1280 --out metrics/pred_yolo26n_fp32.json
"""
import argparse
import json
import os

from common import REPO_ROOT, list_images
from ultralytics import YOLO


def run(model_path, images_dir, imgsz, conf, batch, device, out_path):
    model = YOLO(model_path)
    files = list_images(images_dir)
    if not files:
        raise SystemExit(f"No images found in {images_dir}")

    predictions = []
    for start in range(0, len(files), batch):
        chunk = files[start:start + batch]
        results = model.predict(
            chunk, imgsz=imgsz, conf=conf, device=device, verbose=False, save=False,
        )
        for offset, r in enumerate(results):
            img_id = start + offset
            boxes = r.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls_id in zip(xyxy, conf_arr, cls_arr):
                predictions.append({
                    "image_id": img_id,
                    "category_id": int(cls_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
        print(f"{min(start + batch, len(files))}/{len(files)}", end="\r")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"\n{len(predictions)} detections -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001, help="low conf floor so eval_stratified can sweep thresholds")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "metrics", "pred.json"))
    args = ap.parse_args()
    run(args.model, args.images_dir, args.imgsz, args.conf, args.batch, args.device, args.out)
