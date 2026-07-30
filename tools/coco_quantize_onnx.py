"""ModelOpt ONNX PTQ for the COCO pilot — INT8 arm + FP16 AutoCast arm.

Replicates EXACTLY what ultralytics 8.4.95 does on TensorRT 11
(ultralytics/utils/export/engine.py::modelopt_quantize_onnx), so the pilot's
INT8 engines match the paper pipeline's INT8 arm:
  - modelopt.onnx.quantization.quantize(quantize_mode="int8",
      calibration_method="max", calibration_eps=["cpu"],
      op_types_to_exclude=["Sigmoid"])
  - FP16 via modelopt.onnx.autocast convert_to_mixed_precision
The only differences: calibration images are letterboxed the same way the
engines see val images at inference (Ultralytics uses its dataloader's
letterbox too), and the calib set is our fixed 512-image COCO list.

Usage:
    python tools/coco_quantize_onnx.py --onnx exports/coco_pilot/yolo11n.onnx --mode int8
    python tools/coco_quantize_onnx.py --onnx exports/coco_pilot/yolo11n.onnx --mode fp16
"""
import argparse
import glob
import os

import numpy as np
import onnx

from coco_common import COCO_CALIB_DIR, preprocess


def calib_tensor(imgsz, limit):
    files = sorted(glob.glob(os.path.join(COCO_CALIB_DIR, "*.jpg")))[:limit]
    if not files:
        raise SystemExit(f"no calibration images in {COCO_CALIB_DIR} — run coco_calib_sample.py first")
    mats = [preprocess(f, imgsz)[0][0] for f in files]
    arr = np.stack(mats).astype(np.float32)
    print(f"calibration tensor: {arr.shape} from {len(files)} images")
    return arr


def main(onnx_path, mode, imgsz, limit):
    input_name = onnx.load(onnx_path, load_external_data=False).graph.input[0].name
    if mode in ("int8", "int8_hp32", "fp8"):
        from modelopt.onnx.quantization import quantize as modelopt_quantize
        out = onnx_path.replace(".onnx", f"_{mode}.onnx")
        kwargs = {}
        if mode == "int8_hp32":  # JetPack-6 / TRT 10.3 parser needs fp32 high-precision segments
            kwargs["high_precision_dtype"] = "fp32"
        modelopt_quantize(
            onnx_path,
            quantize_mode="fp8" if mode == "fp8" else "int8",
            calibration_data={input_name: calib_tensor(imgsz, limit)},
            calibration_method="max",
            calibration_eps=["cpu"],
            output_path=out,
            op_types_to_exclude=["Sigmoid"],
            **kwargs,
        )
    else:
        import modelopt.onnx.autocast as autocast
        out = onnx_path.replace(".onnx", "_fp16.onnx")
        onnx.save(
            autocast.convert_to_mixed_precision(
                onnx_path,
                low_precision_type="fp16",
                keep_io_types=True,
                calibration_data={input_name: np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)},
            ),
            out,
        )
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mode", choices=["int8", "int8_hp32", "fp8", "fp16"], required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=512)
    args = ap.parse_args()
    main(args.onnx, args.mode, args.imgsz, args.limit)
