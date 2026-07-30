#!/bin/bash
# Multi-seed replication for the two TT100K detectors that straddle the ~20M
# capacity threshold, where the paper's central claim lives.
#
# Rationale: every detector in the study is a single training run, so a
# referee can ask whether the sign of DIFF at the threshold is a seed
# artifact. Re-training yolo11m (20.2M) and yolo11l (25.5M) under two further
# seeds turns "one checkpoint" into "an effect that survives training noise",
# and yields a seed-to-seed standard deviation that can be compared against
# the size effect itself.
#
# Everything except `seed` is byte-identical to the original recipe:
#   imgsz=1280 epochs=100 batch=8 patience=40 close_mosaic=20 lr0=0.01
#   optimizer=auto deterministic=true            (args.yaml of the seed-0 runs)
#
# Order is interleaved (m,l,m,l) so that an interruption still leaves both
# models with an equal number of seeds.
#
#   setsid nohup bash tools/seed_queue.sh > run_seeds.log 2>&1 < /dev/null &
set -uo pipefail
cd /home/thuan/traffic
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd

# ---- wait for the GPU itself to go idle -----------------------------------
# Matching on script names is unsafe: a finished job can leave a wrapper shell
# whose command line still contains the script name, and the wait then never
# ends. Ask the GPU instead, and require several consecutive idle samples so a
# brief gap between two stages is not mistaken for the end of the queue.
echo "[seeds] waiting for the GPU to go idle..."
idle=0
while [ "$idle" -lt 3 ]; do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed "/^\s*$/d" | wc -l)
  if [ "$n" -eq 0 ]; then idle=$((idle + 1)); else idle=0; fi
  sleep 20
done
echo "[seeds] GPU idle at $(date '+%F %T'); starting"

train_and_ladder() {
  local base=$1 seed=$2
  local name="${base}_s${seed}"
  local ckpt="runs/detect/runs/${name}_fp32/weights/best.pt"

  if [ ! -f "$ckpt" ]; then
    echo "=== TRAIN ${name} (seed ${seed}) START $(date '+%F %T') ==="
    yolo train data=TT100K.yaml model=${base}.pt imgsz=1280 epochs=100 batch=8 \
      project=runs name=${name}_fp32 patience=40 close_mosaic=20 device=0 seed=${seed}
    echo "=== TRAIN ${name} DONE $(date '+%F %T') ==="
  else
    echo "=== TRAIN ${name} already present, skipping ==="
  fi

  if [ -f "$ckpt" ] && [ ! -f "metrics/${name}_fp8.json" ]; then
    echo "=== LADDER ${name} START $(date '+%F %T') ==="
    bash tools/model_ladder.sh "${name}" || echo "LADDER_FAIL ${name}"
  fi
}

train_and_ladder yolo11m 1
train_and_ladder yolo11l 1
train_and_ladder yolo11m 2
train_and_ladder yolo11l 2

echo "=== SEED VARIANCE SUMMARY $(date '+%F %T') ==="
python3 - <<'PYEOF'
import itertools, json, os, statistics as st

def diff(model):
    """DIFF = dAP(S) - dAP(XL) under INT8, on the height strata."""
    try:
        f = json.load(open(f"metrics/{model}_fp32.json"))
        q = json.load(open(f"metrics/{model}_int8_ptq.json"))
    except FileNotFoundError:
        return None
    h = lambda d, b: d["height_bin_ap"][b]["mAP50-95"]
    return ((h(f, "S") - h(q, "S")) - (h(f, "XL") - h(q, "XL")),
            f["overall"]["mAP50-95"], q["overall"]["mAP50-95"])

out = {}
for base in ("yolo11m", "yolo11l"):
    rows = []
    for tag, name in [("seed0", base)] + [(f"seed{s}", f"{base}_s{s}") for s in (1, 2)]:
        r = diff(name)
        if r:
            rows.append((tag, *r))
            print(f"{base} {tag}: FP32={r[1]:.4f} INT8={r[2]:.4f} DIFF={r[0]:+.4f}")
    if len(rows) >= 2:
        ds = [r[1] for r in rows]
        sd = st.stdev(ds) if len(ds) > 2 else abs(ds[0] - ds[1]) / 2
        print(f"  -> {base}: mean DIFF={st.mean(ds):+.4f}, seed spread={sd:.4f}, "
              f"signs={'consistent' if all(d > 0 for d in ds) or all(d < 0 for d in ds) else 'INCONSISTENT'}")
        out[base] = {"per_seed": {r[0]: r[1] for r in rows},
                     "mean": st.mean(ds), "seed_sd": sd,
                     "sign_consistent": all(d > 0 for d in ds) or all(d < 0 for d in ds)}
json.dump(out, open("metrics/seed_variance.json", "w"), indent=1)
print("-> metrics/seed_variance.json")
PYEOF
echo "SEED_QUEUE_ALL_DONE $(date '+%F %T')"
