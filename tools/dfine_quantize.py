"""ModelOpt ONNX PTQ for D-FINE (two-input graph: images + orig_target_sizes).

Same recipe as the YOLO arm (README_journal v2 §3.3): calibration_method=max,
CPU EP, Sigmoid excluded, fixed 512-image calibration list — with the second
input fed the REAL original sizes of the calibration images so the embedded
box-scaling path sees deployment-realistic ranges. D-FINE preprocessing is a
plain bilinear resize (no letterbox), matching its own val pipeline.

Usage:
    python tools/dfine_quantize.py --onnx exports/coco_pilot/dfine_s_explicit.onnx --mode int8
"""
import argparse
import glob
import os

import numpy as np
import onnx
from PIL import Image

from coco_common import COCO_CALIB_DIR


def calib_arrays(imgsz, limit):
    files = sorted(glob.glob(os.path.join(COCO_CALIB_DIR, "*.jpg")))[:limit]
    if not files:
        raise SystemExit(f"no calibration images in {COCO_CALIB_DIR}")
    imgs, sizes = [], []
    for f in files:
        im = Image.open(f).convert("RGB")
        w0, h0 = im.size
        im = im.resize((imgsz, imgsz), Image.BILINEAR)
        imgs.append(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        sizes.append([w0, h0])
    X = np.ascontiguousarray(np.stack(imgs))
    S = np.asarray(sizes, dtype=np.int64)
    print(f"calibration: images {X.shape}, orig_target_sizes {S.shape}")
    return X, S


def main(onnx_path, mode, imgsz, limit):
    from modelopt.onnx.quantization import quantize as modelopt_quantize
    g = onnx.load(onnx_path, load_external_data=False).graph
    in_names = [i.name for i in g.input]
    X, S = calib_arrays(imgsz, limit)
    out = onnx_path.replace(".onnx", f"_{mode}.onnx")
    modelopt_quantize(
        onnx_path,
        quantize_mode=mode,
        calibration_data={in_names[0]: X, in_names[1]: S},
        calibration_method="max",
        calibration_eps=["cpu"],
        output_path=out,
        op_types_to_exclude=["Sigmoid"],
    )
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mode", choices=["int8", "fp8"], required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=512)
    args = ap.parse_args()
    main(args.onnx, args.mode, args.imgsz, args.limit)
