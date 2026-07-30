"""Decoder/pipeline control: Ultralytics-native inference vs the shared engine
pipeline, evaluated by the SAME harness, decomposed by size stratum.

Answers the referee question behind the FP32 fidelity gate: the shared
square-letterbox + replicated-NMS pipeline scores ~0.6-1.0 point below the
officially reported mAP for the classic YOLO heads. Is that deficit
size-uniform (harmless for a size-stratified difference) or concentrated in
the small stratum (which would bias the reference rung of DIFF)?

Runs the PyTorch checkpoint through Ultralytics' own predict/postprocess on
the same val2017 images, writes COCO-format predictions, and evaluates them
with tools/coco_eval_pilot.py. Compare the resulting AP_S/M/L against the
engine-pipeline numbers in metrics/coco_5090/{model}_fp32.json.

Usage (py310 env: torch + ultralytics):
    python tools/coco_decoder_control.py --model yolo11m --limit 0
"""
import argparse
import json
import os

from coco_common import COCO80_TO_91, PILOT_METRICS, val_images


def main(model_name, imgsz, conf, limit, out_path):
    from ultralytics import YOLO

    model = YOLO(f"{model_name}.pt")
    images = val_images()
    if limit:
        images = images[:limit]
    preds = []
    B = 16
    for start in range(0, len(images), B):
        chunk = images[start:start + B]
        results = model.predict([p for _, p in chunk], imgsz=imgsz, conf=conf,
                                iou=0.7, max_det=300, verbose=False, save=False)
        for (img_id, _), r in zip(chunk, results):
            b = r.boxes
            if b is None:
                continue
            for (x1, y1, x2, y2), sc, cl in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist()):
                preds.append({"image_id": int(img_id),
                              "category_id": COCO80_TO_91[int(cl)],
                              "bbox": [x1, y1, x2 - x1, y2 - y1],
                              "score": float(sc)})
        if (start + B) % 800 == 0:
            print(f"{min(start + B, len(images))}/{len(images)}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f)
    print(f"{len(preds)} detections -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(PILOT_METRICS, f"pred_{a.model}_ultralytics.json")
    main(a.model, a.imgsz, a.conf, a.limit, out)
