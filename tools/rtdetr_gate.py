#!/usr/bin/env python3
"""Coordinate-convention probe and FP32 fidelity gate for RT-DETR.

Two things must be settled before any quantized RT-DETR number enters the
study.

*Which preprocessing convention the export assumed.* Ultralytics can reach
640x640 by letterboxing or by stretching, and the two recover different
original-image boxes from the same normalised output. The wrong one does not
degrade gracefully - it collapses AP - so a 500-image probe separates them
cleanly.

*Whether the FP32 engine reproduces the framework.* This is the gate that
caught TensorRT miscompiling D-FINE's deformable attention at full precision,
where the loss would otherwise have been charged to quantization. It runs on
the full validation set against a full-set reference, because AP on a subset
is not comparable to AP on all of it.

Exit status is non-zero if either step fails, which stops the queue before it
can produce a number that looks like quantization damage but is not.
"""
import argparse, json, os, subprocess, sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
M = os.path.join(R, "metrics", "coco_pilot")
E = os.path.join(R, "exports", "coco_pilot")
TOOLS = os.path.join(R, "tools")


def run_eval(model, mode, tag, limit):
    """Infer + evaluate; returns AP or None. limit=0 means the whole set."""
    pred = os.path.join(M, f"pred_{tag}.json")
    cmd = [PY, os.path.join(TOOLS, "coco_infer_trt.py"),
           "--engine", os.path.join(E, f"{model}.plan"),
           "--decoder", "rtdetr", "--rtdetr-mode", mode, "--out", pred]
    if limit:
        cmd += ["--limit", str(limit)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    inference failed ({mode}):\n{r.stderr[-400:]}", flush=True)
        return None
    cmd = [PY, os.path.join(TOOLS, "coco_eval_pilot.py"), "--dt", pred,
           "--model", tag, "--precision", "fp32"]
    if limit:
        cmd += ["--img-ids-from-dt"]      # score only the images actually run
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    evaluation failed ({mode}):\n{r.stderr[-400:]}", flush=True)
        return None
    try:
        return json.load(open(os.path.join(M, f"{tag}_fp32.json")))["stats"]["AP"]
    except Exception as e:
        print(f"    could not read AP: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--probe", type=int, default=500)
    a = ap.parse_args()

    ref_path = os.path.join(M, "rtdetr_torch_reference.json")
    if not os.path.exists(ref_path):
        print("no framework reference; cannot gate"); return 1
    ref = json.load(open(ref_path))

    chosen, failed, report = {}, [], {}
    for m in a.models:
        target = ref.get(m.replace("_", "-"), {}).get("AP")
        print(f"\n{m}: framework reference AP (full val) = "
              f"{'unknown' if target is None else round(target, 4)}", flush=True)
        if target is None:
            failed.append(m); continue

        print(f"  probing the coordinate convention on {a.probe} images", flush=True)
        probe = {}
        for mode in ("stretch", "letterbox"):
            ap_v = run_eval(m, mode, f"probe_{m}_{mode}", a.probe)
            probe[mode] = ap_v
            print(f"    {mode:10s} AP = {'n/a' if ap_v is None else round(ap_v, 4)}",
                  flush=True)
        ok = {k: v for k, v in probe.items() if v is not None}
        if not ok:
            failed.append(m); continue
        mode = max(ok, key=ok.get)
        other = [v for k, v in ok.items() if k != mode]
        # A wrong convention collapses AP; if the two are close, neither is
        # obviously right and the probe has not actually decided anything.
        if other and ok[mode] - other[0] < 0.05:
            print(f"    conventions are within 0.05 AP of each other - the probe "
                  f"cannot separate them, so the export is not understood")
            failed.append(m); report[m] = {"probe": probe, "verdict": "ambiguous"}
            continue
        print(f"    -> {mode}", flush=True)

        print("  full-set FP32 fidelity gate", flush=True)
        full = run_eval(m, mode, m, 0)
        if full is None:
            failed.append(m); continue
        delta = abs(full - target)
        print(f"    engine AP = {full:.4f} | reference {target:.4f} | "
              f"delta {delta:.4f} | tol {a.tol}", flush=True)
        report[m] = {"probe": probe, "mode": mode, "engine_AP": full,
                     "reference_AP": target, "delta": delta}
        if delta > a.tol:
            print("    GATE FAIL", flush=True); failed.append(m)
        else:
            print("    GATE PASS", flush=True); chosen[m] = mode

    json.dump({"chosen_mode": chosen, "failed": failed, "tol": a.tol,
               "probe_images": a.probe, "detail": report},
              open(os.path.join(M, "rtdetr_gate.json"), "w"), indent=1)
    if failed:
        print(f"\nGate failed for: {failed}")
        print("A mismatch here means the export or the decode is wrong, not that "
              "RT-DETR is inaccurate. No quantized number should be reported.")
        return 1
    print(f"\nAll models passed. Chosen conventions: {chosen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
