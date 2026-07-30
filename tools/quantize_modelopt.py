"""INT8-QAT / FP8-PTQ / FP4(NVFP4)-PTQ via NVIDIA ModelOpt (README §7.2-7.3).

IMPORTANT — verified against the installed nvidia-modelopt==0.45.0, not
against README's original snippet, which is stale on two points:
  1. `modelopt.torch.opt.export()` does not exist in this version. The
     working pattern is: mtq.quantize(...) registers custom ONNX symbolic
     ops as a side effect, so a PLAIN `torch.onnx.export(...)` afterwards
     already emits the right Q/DQ nodes — no separate "mto.export" call.
  2. `python -m modelopt.onnx.quantization --quantize_mode nvfp4` does NOT
     work — this CLI's --quantize_mode only accepts {fp8, int8, int4} in
     0.45.0. NVFP4 is only exposed via the torch-native path used here
     (mtq.NVFP4_DEFAULT_CFG), not the ONNX-PTQ CLI README's §7.3 shows.

UNVERIFIED ON REAL HARDWARE — needs a free GPU to test, two specific risks:
  - --finetune-epochs (QAT) reassigns yolo.model then calls yolo.train();
    Ultralytics' trainer may reload/rebuild the model internally instead of
    reusing the already-quantized module. Verify with a 1-epoch dry run
    before trusting a real QAT budget.
  - The exported ONNX comes from a raw torch.onnx.export on the underlying
    DetectionModel, bypassing Ultralytics' own exporter (engine/exporter.py)
    which does extra output post-processing/metadata embedding. The
    resulting engine's output tensor layout may NOT match what
    tools/infer.py (built around ultralytics' own AutoBackend loading
    convention) expects — check this before trusting fp8/fp4 eval numbers.

Usage (PTQ, no finetune):
    python tools/quantize_modelopt.py --weights best.pt --mode fp8 \
        --calib-dir data/calib/images --imgsz 1280 --out exports/yolo26n_fp8.onnx

Usage (QAT, README §7.2):
    python tools/quantize_modelopt.py --weights best.pt --mode int8 \
        --calib-dir data/calib/images --imgsz 1280 --finetune-epochs 8 \
        --data configs/tt100k.yaml --out exports/yolo26n_int8qat.onnx
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import modelopt.torch.quantization as mtq
from ultralytics import YOLO

MODE_CFG = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
}


def load_calib_batches(calib_dir, imgsz, batch_size, device):
    files = sorted(glob.glob(os.path.join(calib_dir, "*.jpg")))
    if not files:
        raise SystemExit(f"No calibration images found in {calib_dir}")
    batch = []
    for f in files:
        im = cv2.imread(f)
        im = cv2.resize(im, (imgsz, imgsz))
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        batch.append(im)
        if len(batch) == batch_size:
            yield torch.from_numpy(np.ascontiguousarray(batch)).float().to(device) / 255.0
            batch = []
    if batch:
        yield torch.from_numpy(np.ascontiguousarray(batch)).float().to(device) / 255.0


def quantize(weights, mode, calib_dir, imgsz, device, finetune_epochs, data_yaml, out_path):
    yolo = YOLO(weights)
    model = yolo.model.to(device).eval()

    def forward_loop(m):
        with torch.no_grad():
            for batch in load_calib_batches(calib_dir, imgsz, 8, device):
                m(batch)

    model = mtq.quantize(model, MODE_CFG[mode], forward_loop=forward_loop)
    mtq.print_quant_summary(model)

    if finetune_epochs > 0:
        yolo.model = model
        yolo.train(data=data_yaml, imgsz=imgsz, epochs=finetune_epochs, device=device,
                   pretrained=False, project="runs", name=f"{os.path.basename(weights).split('.')[0]}_qat")
        model = yolo.model

    model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Static batch=1 export: modelopt's FP8/FP4 Q/DQ symbolic ops lose shape
    # info through dynamic axes, which then breaks the plain conv exporter
    # ("kernel of unknown shape"). infer.py already standardized on bs=1
    # engines anyway (see its docstring), so this costs nothing.
    torch.onnx.export(model, dummy, out_path, opset_version=17, dynamo=False,
                       input_names=["images"], output_names=["output"])
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mode", choices=list(MODE_CFG), required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--finetune-epochs", type=int, default=0)
    ap.add_argument("--data", default="configs/tt100k.yaml", help="only used if --finetune-epochs > 0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    quantize(args.weights, args.mode, args.calib_dir, args.imgsz, args.device,
              args.finetune_epochs, args.data, args.out)
