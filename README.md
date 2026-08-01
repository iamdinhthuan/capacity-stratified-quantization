# Capacity-Stratified Analysis of INT8 and FP8 Quantization for Object Detection

Code, quantized-graph recipes, calibration lists, pre-registration document and
every per-model metric file behind the paper of the same name.

Detectors are evaluated **as deployed** — official or single-recipe checkpoints
exported to ONNX, quantized once with NVIDIA Model Optimizer, and compiled into
real TensorRT engines per device. Accuracy is then decomposed by object size,
because the aggregate mAP drop that normally gates a quantized model averages
over exactly the structure this paper is about.

Twenty-two checkpoints across five detector families — YOLO11, YOLOv8 and
YOLO26 (convolutional), D-FINE and RT-DETR (detection transformers) — on two
benchmarks (TT100K, COCO val2017) and three measured devices: RTX 5090, RTX
4090 and a Jetson Orin Nano Super. RT-DETR was added post hoc, after D-FINE
collapsed, to test whether that collapse is a property of the architecture
class; it enters no pre-registered test and is marked as such in the paper.

No Jetson AGX Thor device was available, so the FP8 recommendation is measured
on desktop GPUs and extrapolated to that generation rather than confirmed
there.

## What is here

| Path | Contents |
|---|---|
| `tools/` | Everything that produced a number: export, quantization, engine build, inference, stratified evaluation, bootstrap, permutation and equivalence tests, segmented and capacity-slope regressions, latency/power measurement, figure generation |
| `metrics/` | Per-model metric files (TT100K), `metrics/coco_5090/` and `metrics/coco_pilot/` (COCO, canonical and cross-device builds), `metrics/orin/`, `metrics/ada/` (cross-generation latency), plus every bootstrap, TOST and regression output |
| `configs/` | Size-bin definitions, class map, and the **fixed calibration image lists** — the same 512 images used for every quantized graph |
| `PREREGISTRATION_coco_confirmatory.md` | The frozen analysis plan for the confirmatory stage. Read this before the results |
| `environment.md`, `requirements.txt` | The exact stack; latency and power tables are meaningless without it |

Not committed, because they are large or fully regenerable from the above: the
datasets, ONNX graphs, TensorRT engines, training checkpoints and per-image
prediction files. The manuscript is not distributed here either. `tools/`
rebuilds every artifact, figures included (`python tools/coco_make_figs.py`).

## Reproducing a single number

```bash
# quantize one model to INT8 from its exported ONNX graph
python tools/coco_quantize_onnx.py --onnx exports/coco_pilot/yolo11m.onnx --mode int8

# build the engine and run stratified evaluation
python tools/build_engine.py   --onnx exports/coco_pilot/yolo11m_int8.onnx \
                               --engine exports/coco_pilot/yolo11m_int8.plan
python tools/coco_infer_trt.py --engine exports/coco_pilot/yolo11m_int8.plan \
                               --out metrics/coco_pilot/pred_yolo11m_int8.json
python tools/coco_eval_pilot.py --dt metrics/coco_pilot/pred_yolo11m_int8.json \
                                --model yolo11m --precision int8

# paired image-level bootstrap CI on the small-minus-large gap
python tools/coco_boot_diff.py --models yolo11m --n-boot 2000
```

The analyses that carry the paper's claims run straight off the committed
metric files, with no GPU required:

```bash
python tools/coco_slope_test.py     # capacity slope, three nested model sets
python tools/coco_slope_robust.py   # leave-one-out, leave-one-family-out, rank statistic
python tools/coco_segmented.py      # breakpoint with a Davies-safe permutation test
python tools/tt100k_slope.py        # the same slope on the controlled-training study
python tools/coco_confirmatory_test.py --spec ...   # the pre-registered permutation test
```

Scripts that reach other machines take the host from the environment
(`GPU_HOST`, `ORIN_HOST`) rather than hard-coding one.

## Reading the results honestly

Three things a reader should know without having to dig for them.

**The pre-registered test was not supported.** The registered hypothesis — that
detectors above 20M parameters show a larger small-minus-large degradation gap
than those below — returns *p* = 0.55 on its primary set. That verdict is
reported unchanged in the paper and is not walked back here. The reason is
identified rather than explained away: the primary set spans two architecture
classes that share no regime, and the D-FINE family collapses under uniform
post-training quantization at every capacity.

**The continuous capacity analysis is post hoc.** Regressing the gap on
log-parameters with lineage fixed effects — same data, same within-family
randomisation, different statistic — was specified after the registered tests
had run. It is labelled post hoc everywhere it appears. It is *also* null on the
registered primary set (*p* = 0.24); it is decisive only within the fifteen
convolutional models (+0.0495 AP per decade, *p* < 10⁻⁴).

**Quantized engine accuracy varies across TensorRT builds.** Identical Q/DQ
graphs land up to 1.0 mAP point apart across TensorRT generations, while FP32
and FP16 reproduce to within 10⁻³. Per-model intervals are therefore presented
as estimation rather than as a family of tests, and one canonical device is
fixed per table.

## Datasets and checkpoints

TT100K and COCO are public benchmarks and are not redistributed here. Every
quantized detector derives from a publicly released checkpoint of its
respective project.

## Citation

Please cite the article. The manuscript itself is not distributed here; this
repository holds the code and data needed to check and reproduce its numbers.
