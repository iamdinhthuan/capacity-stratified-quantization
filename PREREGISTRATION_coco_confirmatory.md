# Pre-registration — Confirmatory capacity contrast on COCO (INT8 size-unfairness)

**Frozen at: 2026-07-28 03:05 (+07), BEFORE any confirmatory-family model was evaluated on COCO.**
*(Ghi chú tiếng Việt: file này phải được commit vào git TRƯỚC khi chạy YOLOv8/D-FINE/YOLO26 trên COCO — anh commit giùm em sáng nay, hoặc đẩy lên OSF. Không sửa nội dung sau khi confirmatory bắt đầu; mọi thay đổi = amendment có ghi ngày.)*

## 1. Background (hypothesis-generating data, already collected)
- TT100K (conference study, 7 models): INT8 small-vs-large degradation gap significant only in the three ≥20M YOLO11 variants; sub-10M variants indistinguishable from size-neutral.
- COCO exploratory sweep (YOLO11 family only, completed 2026-07-28 ~02:50): DIFF = ΔAP_small − ΔAP_large under INT8 was **monotone in capacity and sign-flipping**: n(2.6M) −0.0236 [95% CI −0.0369,−0.0079]; s(9.4M) −0.0116 [−0.0225,−0.0002]; m(20.1M) +0.0080; l(25.3M) +0.0100; x(56.9M) +0.0161 (m/l/x CIs pending at freeze time; point estimates recorded).
- No YOLOv8, D-FINE, or YOLO26 model has been evaluated on COCO under this protocol as of the freeze timestamp.

## 2. Confirmatory hypothesis (H1, one-sided)
Across the confirmatory model set, the mean DIFF of models with ≥20M parameters is **greater** than the mean DIFF of models with <20M parameters, under INT8 post-training quantization on COCO val2017.
- DIFF = [AP_small(FP32-engine) − AP_small(INT8)] − [AP_large(FP32-engine) − AP_large(INT8)], mAP@[.5:.95], standard COCO mask-area strata, maxDets=[1,10,100], pycocotools.
- Threshold 20M and direction were fixed from the hypothesis-generating data above.

## 3. Confirmatory model set (official COCO checkpoints, as released)
- Primary: YOLOv8 n/s/m/l/x (3.2/11.2/25.9/43.7/68.2M); D-FINE N/S/M/L/X (4/10/19/31/62M).
  - Group split at 20M → low = {v8n, v8s, DF-N, DF-S, DF-M(19M)}, high = {v8m, v8l, v8x, DF-L, DF-X}.
- Secondary (reported, not in primary test): YOLO26 n/s/m/l/x, end-to-end (NMS-free) mode.
- YOLO11 results remain exploratory (they generated the hypothesis) and are excluded from the primary test.

## 4. Primary analysis
- One-sided exact permutation test, **stratified by family** (permute the low/high labels within each family; enumerate all rearrangements), statistic = difference of group means of DIFF; α = 0.05.
- Robustness check: one-sided Welch's t on the same groups.

## 5. Secondary analyses (pre-specified)
- Per-model paired image-level bootstrap, B = 2,000, BCa intervals for DIFF.
- TOST equivalence for low-capacity models: size-neutrality claimed only if the 90% CI of DIFF lies within [−Δ, +Δ] with **Δ = 0.005** (0.5 AP point; smallest deployment-relevant differential degradation). Sensitivity at 1.5Δ and 2Δ.
- Kendall's τ of DIFF vs log(params) across all confirmatory models (descriptive), and segmented regression estimating the capacity breakpoint with bootstrap CI (descriptive check against the frozen 20M, not a replacement).
- Holm–Bonferroni across the two co-primary datasets (COCO confirmatory contrast; TT100K case-study contrast).

## 6. Frozen pipeline (identical to exploratory stage)
- ONNX from official checkpoints (Ultralytics export 640, batch 1, opset 17; D-FINE export_onnx.py with dfine-cpp gather-bilinear rewrite + FP16-parity gate before any quantized number is used).
- ModelOpt 0.45.0 ONNX PTQ: quantize_mode=int8, calibration_method=max, calibration_eps=cpu, op_types_to_exclude=[Sigmoid], fixed 512-image calibration set (data/coco/calib/calib_list.json, seed 0).
- Strongly-typed TensorRT engines; FP32-engine reference rung; one shared decoder (conf > 1e-3, per-class NMS IoU 0.7, top-300, clip); letterbox 640.
- Eval: pycocotools on instances_val2017.json, maxDets=[1,10,100]; per-run FP16 pipeline-integrity gate: |AP(FP16) − AP(FP32)| ≤ 0.002 required before INT8/FP8 numbers are accepted.

## 7. Exclusion & failure handling
- A model whose INT8 engine fails to build after the documented fallbacks (hp32 flavor; weak-typed int8+fp16 on TRT ≤10) is reported as a build failure and excluded from the primary test (with the failure itself reported as a finding).
- No other exclusions permitted. No peeking at confirmatory DIFF values before the full set completes.

## 8. Interpretation rules
- H1 supported: permutation p < 0.05 → the capacity-dependent size-unfairness generalizes across architecture families on COCO.
- H1 not supported: report honestly as a confirmatory failure; the exploratory sign-flip remains a YOLO11-specific observation; no post-hoc threshold search will be presented as confirmatory.

*Registered by: automated overnight run under the direction of the paper's first author. Data custodian note: exploratory artifacts under metrics/coco_pilot/ at freeze time cover only yolo11n/s/m/l/x.*
