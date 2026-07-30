"""FP16 / INT8-PTQ TensorRT engine export, and plain ONNX export, via Ultralytics
(README §7.1, §7.3 step 1). Needs a free GPU — do not run while another
training/inference job is using most of the VRAM.

INT8 calibration uses the fixed, size-balanced calib set from
tools/build_calib.py (configs/tt100k_calib.yaml) instead of Ultralytics'
default random `fraction` sampling, so every precision on the ladder that
needs calibration shares the exact same images (README §7.4).

Usage:
    python tools/export_engine.py --weights runs/detect/runs/yolo26n_fp32/weights/best.pt --mode onnx
    python tools/export_engine.py --weights ... --mode fp16
    python tools/export_engine.py --weights ... --mode int8 --calib-yaml configs/tt100k_calib.yaml
"""
import argparse
import os

from ultralytics import YOLO


def export(weights, mode, imgsz, calib_yaml, device):
    model = YOLO(weights)
    kwargs = dict(imgsz=imgsz, device=device)
    if mode == "onnx":
        kwargs.update(format="onnx", opset=17, simplify=True)
    elif mode == "fp16":
        kwargs.update(format="engine", half=True)
    elif mode == "int8":
        kwargs.update(format="engine", int8=True, data=calib_yaml, fraction=1.0)
    else:
        raise ValueError(mode)
    path = model.export(**kwargs)
    # Ultralytics derives the engine name from the weights (best.engine) for BOTH
    # fp16 and int8, so a later export silently overwrites the earlier one. Give
    # each precision its own file so a benchmark can't be run on the wrong engine.
    if mode in ("fp16", "int8") and os.path.exists(path):
        tagged = f"{os.path.splitext(path)[0]}.{mode}.engine"
        os.replace(path, tagged)
        path = tagged
    print(f"{mode} -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mode", choices=["onnx", "fp16", "int8"], required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--calib-yaml", default="configs/tt100k_calib.yaml")
    ap.add_argument("--device", default=0)
    args = ap.parse_args()
    export(args.weights, args.mode, args.imgsz, args.calib_yaml, args.device)
