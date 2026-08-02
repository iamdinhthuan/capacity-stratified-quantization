# Size-Stratified Evaluation of INT8 and FP8 Quantization for Object Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21754190.svg)](https://doi.org/10.5281/zenodo.21754190)

Code, quantized-graph recipes, calibration lists, the frozen analysis plan and
every per-model metric file behind the paper of the same name.

The plan was frozen prospectively but was never deposited with a third-party
registry, so its timeline is attested by the authors and by file history, not
by an external timestamp. "Registered" throughout means *named in that frozen
plan*.

Detectors are evaluated **as deployed** — official checkpoints, or checkpoints trained here under one recipe,
exported to ONNX, quantized once with NVIDIA Model Optimizer, and compiled into
real TensorRT engines per device. Accuracy is then decomposed by object size,
because the aggregate mAP drop that normally gates a quantized model averages
over exactly the structure this paper is about.

Twenty-two checkpoints across five detector families — YOLO11, YOLOv8 and
YOLO26 (convolutional), D-FINE and RT-DETR (detection transformers) — on two
benchmarks (TT100K, COCO val2017) and three measured devices: RTX 5090, RTX
4090 and a Jetson Orin Nano Super. RT-DETR was added post hoc, after D-FINE
collapsed, to test whether that collapse is a property of the architecture
class; it enters no registered test and is marked as such in the paper.

No Jetson AGX Thor device was available, so the FP8 recommendation is measured
on desktop GPUs and extrapolated to that generation rather than confirmed
there.

## Citing this repository

Archived on Zenodo. The concept DOI always resolves to the current version:

```
10.5281/zenodo.21754190          all versions
10.5281/zenodo.21754191          v1.0-ivc-submission, the snapshot accompanying the article
```


## What is here

| Path | Contents |
|---|---|
| `tools/` | Everything that produced a number: export, quantization, engine build, inference, stratified evaluation, bootstrap, permutation and equivalence tests, segmented and capacity-slope regressions, latency/power measurement, figure generation |
| `metrics/` | Per-model metric files (TT100K), `metrics/coco_5090/` and `metrics/coco_pilot/` (COCO, canonical and cross-device builds), `metrics/orin/`, `metrics/ada/` (cross-generation latency), plus every bootstrap, TOST and regression output |
| `configs/` | Size-bin definitions, class map, and the **fixed calibration image lists** — the same 512 images used for every quantized graph |
| `PREREGISTRATION_coco_confirmatory.md` | The frozen analysis plan for the confirmatory stage. Read this before the results |
| `LICENSE` | MIT, covering the code and the metric files. The COCO and TT100K datasets and the detector checkpoints keep their own licences and are not redistributed here |
| `environment.md`, `requirements.txt` | The exact stack; latency and power tables are meaningless without it |

Not committed, because they are large or fully regenerable from the above: the
datasets, ONNX graphs, TensorRT engines, training checkpoints and per-image
prediction files. The manuscript is not distributed here either. `tools/`
rebuilds every artifact, figures included (`python tools/coco_make_figs.py`).

## What the later analyses added

Three analyses postdate the first release and are in `metrics/coco_5090/`:

| File(s) | What it settles |
|---|---|
| `*_int8ent.json`, `calibrator_sensitivity.json` | INT8 re-quantized with `entropy` instead of the shipping `max` activation calibrator. Entropy is better on every convolutional checkpoint (loss cut 38-66%), and it removes most of the size structure with it: the capacity slope falls from +0.053 to +0.017, and to +0.001 without the nano rungs. The capacity result therefore belongs to the `max`-calibrated recipe, not to INT8 in general |
| `boot2k_fp8_yolo11_bca.json` | The bias-corrected and accelerated intervals the plan asked for, computed for the YOLO11 FP8 arm with a leave-one-out jackknife over all 5,000 images. They are stricter than the percentile intervals reported first |
| `sub500_dfine_s_int8selbb.json`, `ortdisc_*.json` | TensorRT against ONNX Runtime on the same 500 images for the backbone-quantized D-FINE-S graph: 0.2203 AP against 0.0039, with an ONNX Runtime FP32 control of 0.5160 |

`tools/review_experiments2.sh`, `tools/entropy_finish.sh` and
`tools/bca_fleet.sh` are the drivers for these three.


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
