# Environment (locked)

Every training / quantization / evaluation number in this repo was produced on
one machine, under one conda env. Recorded here because latency/power tables are
meaningless without the exact stack.

## Hardware
- **GPU:** 1× NVIDIA GeForce RTX 5090 (Blackwell, `sm_120`, 32 GB), confirmed via
  `torch.cuda.get_device_capability()` → `(12, 0)`.
- Driver `610.43.02`, system CUDA `13.3`.
- Shared multi-user box; system-metric measurements assert an idle GPU and sample
  the SM clock (no root → no `nvidia-smi -lgc` clock lock; stability is reported
  instead — see `tools/measure_system.py`).

## Software (conda env `qtsd`, Python 3.11)
```
torch            2.11.0+cu128
torchvision      0.26.0+cu128
ultralytics      8.4.95
tensorrt         11.1.0.106      (pip wheel; ships sm_120 kernels)
nvidia-modelopt  0.45.0
onnx             1.21.0
onnxruntime-gpu  1.24.4
onnxsim          0.6.5
pycocotools      2.0.11
numpy            2.4.4
scipy            1.17.1
pandas           3.0.3
matplotlib       3.11.0
seaborn          0.13.2
opencv-python    5.0.0
pyyaml           6.0.3
```

## Toolchain caveats discovered the hard way (all real, all verified on hardware)
- `nvidia-modelopt==0.45.0`: `modelopt.torch.opt.export()` does **not** exist;
  `torch.onnx.export(...)` needs `dynamo=False` for its Q/DQ symbolic ops. The
  ONNX-PTQ CLI (`python -m modelopt.onnx.quantization`) supports `--quantize_mode
  {fp8,int8,int4}` only — **no `nvfp4`**. INT8-QAT is blocked by a CUDA-extension
  ABI failure against torch 2.11 headers (CPU fallback segfaults). So the working
  ladder is FP32 → FP16 → INT8-PTQ → FP8; FP4 and QAT are out of scope in this
  toolchain (documented, not hidden).
- `trtexec` binary is **not** in the pip `tensorrt` wheel (Python bindings + `.so`
  only). `tools/build_engine.py` builds engines through the TensorRT Python API
  instead; a real `trtexec` from the SDK tar is optional (`tools/trtexec_env.sh`).
- Ultralytics exports TensorRT engines with a **static batch size of 1** unless
  `dynamic=True`; `tools/infer.py` therefore defaults to `--batch 1`.
- RT-DETR through Ultralytics→TensorRT emits `(1, 300, 6)` rows of *normalised*
  `cxcywh` plus an explicit score and class, which neither decoder in
  `coco_common.py` handles — an early note in this file recorded that as
  "postprocesses incorrectly, excluded from the paper". It is not excluded.
  `tools/rtdetr_decode.py` decodes that layout, and `tools/rtdetr_gate.py`
  settles the coordinate convention against a 500-image probe before checking
  the FP32 engine against the framework's own validation AP on the full set.
  Both RT-DETR checkpoints pass that gate (deltas 0.0077 and 0.0086) and are
  reported in the paper as a post-hoc addition.
