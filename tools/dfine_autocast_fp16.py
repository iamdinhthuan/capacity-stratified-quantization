"""ModelOpt AutoCast FP16 for the 2-input D-FINE deployment ONNX.

Same call as tools/coco_quantize_onnx.py --mode fp16 (modelopt.onnx.autocast
convert_to_mixed_precision, keep_io_types=True), but coco_quantize_onnx.py
assumes a single image input; D-FINE has a second int64 input
`orig_target_sizes`, so calibration_data must carry both tensors.

Usage (venv_pilot):
    python dfine_autocast_fp16.py --onnx onnx/dfine_n.onnx   # -> onnx/dfine_n_fp16.onnx
"""
import argparse

import numpy as np
import onnx
import modelopt.onnx.autocast as autocast


def main(onnx_path, imgsz):
    out = onnx_path.replace(".onnx", "_fp16.onnx")
    calib = {
        "images": np.random.randn(1, 3, imgsz, imgsz).astype(np.float32),
        "orig_target_sizes": np.array([[imgsz, imgsz]], dtype=np.int64),
    }
    m = autocast.convert_to_mixed_precision(
        onnx_path,
        low_precision_type="fp16",
        keep_io_types=True,
        calibration_data=calib,
    )
    onnx.save(m, out)
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    main(args.onnx, args.imgsz)
