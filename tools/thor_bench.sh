#!/usr/bin/env bash
# =====================================================================
#  Jetson AGX Thor benchmark - run this ON the Thor itself
#  (no network, no SSH, no Python packages beyond the system python3)
#
#  USAGE
#     1. copy the whole folder onto the Thor (USB stick is fine)
#     2. open a terminal in this folder
#     3. (recommended, needs the sudo password once)
#            sudo nvpmodel -m 0
#            sudo jetson_clocks
#     4. bash thor_bench.sh
#     5. when it finishes, send back the single file:  thor_results.tar.gz
#
#  It is safe to stop it (Ctrl-C) and run it again - finished work is kept.
#  Options:  MODELS="yolo11n yolo11m yolo11x" bash thor_bench.sh    (shorter run)
#            DURATION=10 bash thor_bench.sh                          (quick test)
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

MODELS=${MODELS:-"yolo11n yolo11s yolo11m yolo11l yolo11x"}
PRECS=${PRECS:-"fp32 fp16 int8 fp8"}
DURATION=${DURATION:-30}          # seconds of timed inference per engine
WARMUP=${WARMUP:-1000}            # ms
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

OUT=results
ENG=engines
LOG=$OUT/logs
mkdir -p "$OUT" "$ENG" "$LOG"
JSONL=$OUT/thor_bench.jsonl
touch "$JSONL"

say() { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "!! $*" | tee -a "$OUT/problems.txt"; }

# ---------------------------------------------------------------- checks
say "checking the environment"
[ "$(uname -m)" = "aarch64" ] || fail "this does not look like a Jetson (uname -m = $(uname -m))"
if [ ! -x "$TRTEXEC" ]; then
  fail "trtexec not found at $TRTEXEC"
  echo "   try:  sudo apt-get install -y tensorrt"
  echo "   or set the path:  TRTEXEC=/path/to/trtexec bash thor_bench.sh"
  exit 1
fi
ls onnx/*.onnx >/dev/null 2>&1 || { fail "no ONNX graphs found in ./onnx - is the folder complete?"; exit 1; }

# ---------------------------------------------------------------- provenance
{
  echo "date            : $(date -Is)"
  echo "device          : $(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
  echo "uname           : $(uname -a)"
  echo "L4T             : $(head -1 /etc/nv_tegra_release 2>/dev/null)"
  echo "jetpack pkg     : $(dpkg -l 2>/dev/null | awk '/nvidia-jetpack /{print $3; exit}')"
  echo "tensorrt pkg    : $(dpkg -l 2>/dev/null | awk '/libnvinfer[0-9] /{print $3; exit}')"
  echo "trtexec         : $("$TRTEXEC" --version 2>&1 | head -2 | tr '\n' ' ')"
  echo "cuda            : $(nvcc --version 2>/dev/null | tail -1)"
  echo "nvpmodel        : $(nvpmodel -q 2>/dev/null | tr '\n' ' ')"
  echo "clocks_locked   : $(pgrep -x jetson_clocks >/dev/null && echo maybe || echo 'unknown - run sudo jetson_clocks before benchmarking')"
  echo "cpu governor    : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
} | tee "$OUT/environment.txt"
echo

# ---------------------------------------------------------------- helpers
# Thor exposes different power rails from Orin (VDD_GPU, VDD_CPU_SOC_MSS,
# VIN_SYS_5V0 ...), so every "NAME <n>mW" pair is captured generically and
# averaged afterwards rather than assuming one rail name.
sample_power() {                 # $1 = logfile
  tegrastats --interval 200 > "$1" 2>/dev/null &
  echo $!
}

parse_run() {                    # $1 = trtexec log, $2 = tegrastats log
  python3 - "$1" "$2" <<'PY'
import json, re, sys
trt, tegra = sys.argv[1], sys.argv[2]
txt = open(trt, errors="ignore").read()
def g(pat, cast=float):
    m = re.search(pat, txt)
    return cast(m.group(1)) if m else None
row = {
  "throughput_qps": g(r"Throughput:\s*([0-9.]+)\s*qps"),
  "gpu_mean_ms":    g(r"GPU Compute Time:.*?mean = ([0-9.]+) ms"),
  "gpu_median_ms":  g(r"GPU Compute Time:.*?median = ([0-9.]+) ms"),
  "gpu_p90_ms":     g(r"GPU Compute Time:.*?percentile\(90%\) = ([0-9.]+) ms"),
  "gpu_p99_ms":     g(r"GPU Compute Time:.*?percentile\(99%\) = ([0-9.]+) ms"),
  "host_mean_ms":   g(r"Host Latency:.*?mean = ([0-9.]+) ms"),
}
# every "RAIL 1234mW/5678mW" pair, averaged over the steady-state window
rails = {}
lines = open(tegra, errors="ignore").read().splitlines()
for ln in lines[5:]:                       # drop the first ~1s of ramp-up
    for name, inst in re.findall(r"([A-Z][A-Z0-9_]+)\s+(\d+)mW", ln):
        rails.setdefault(name, []).append(int(inst))
row["power_mw"] = {k: round(sum(v)/len(v), 1) for k, v in rails.items() if v}
row["power_samples"] = len(lines)
# thermals, if present
temps = {}
for ln in lines[5:]:
    for name, val in re.findall(r"([a-z_]+)@([0-9.]+)C", ln):
        temps.setdefault(name, []).append(float(val))
row["temp_c_max"] = {k: max(v) for k, v in temps.items()} if temps else {}
print(json.dumps(row))
PY
}

done_already() { grep -q "\"key\": \"$1\"" "$JSONL" 2>/dev/null; }

# ---------------------------------------------------------------- main loop
for M in $MODELS; do
  for P in $PRECS; do
    KEY="${M}_${P}"
    done_already "$KEY" && { say "$KEY already measured, skipping"; continue; }

    # pick the ONNX for this rung; INT8 has a fallback flavour because the
    # JetPack-6 parser rejected the default one (kept here in case Thor's
    # TensorRT does the same)
    case $P in
      fp32) SRC=onnx/${M}.onnx ;;
      fp16) SRC=onnx/${M}_fp16.onnx ;;
      int8) SRC=onnx/${M}_int8.onnx ;;
      fp8)  SRC=onnx/${M}_fp8.onnx ;;
    esac
    [ -f "$SRC" ] || { say "$KEY: no graph ($SRC), skipping"; continue; }

    PLAN=$ENG/${KEY}.plan
    if [ ! -f "$PLAN" ]; then
      say "$KEY: building engine"
      if ! "$TRTEXEC" --onnx="$SRC" --stronglyTyped --saveEngine="$PLAN" \
             --skipInference > "$LOG/build_${KEY}.log" 2>&1; then
        if [ "$P" = "int8" ] && [ -f onnx/${M}_int8_hp32.onnx ]; then
          say "$KEY: default INT8 graph rejected, trying the fp32-high-precision flavour"
          if ! "$TRTEXEC" --onnx=onnx/${M}_int8_hp32.onnx --stronglyTyped \
                 --saveEngine="$PLAN" --skipInference \
                 > "$LOG/build_${KEY}_hp32.log" 2>&1; then
            fail "$KEY: BUILD FAILED (both flavours) - see $LOG/build_${KEY}*.log"
            echo "{\"key\":\"$KEY\",\"model\":\"$M\",\"precision\":\"$P\",\"build\":\"failed\"}" >> "$JSONL"
            continue
          fi
          echo "{\"key\":\"${KEY}_flavour\",\"note\":\"int8 used hp32 flavour\"}" >> "$JSONL"
        else
          fail "$KEY: BUILD FAILED - see $LOG/build_${KEY}.log"
          echo "{\"key\":\"$KEY\",\"model\":\"$M\",\"precision\":\"$P\",\"build\":\"failed\"}" >> "$JSONL"
          # an FP8 build failure on Thor is itself a result worth reporting
          continue
        fi
      fi
    fi

    say "$KEY: timing ${DURATION}s"
    TPID=$(sample_power "$LOG/tegra_${KEY}.log")
    sleep 2
    "$TRTEXEC" --loadEngine="$PLAN" --warmUp=$WARMUP --duration=$DURATION \
        --useSpinWait > "$LOG/bench_${KEY}.log" 2>&1
    RC=$?
    sleep 1
    kill "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null
    if [ $RC -ne 0 ]; then
      fail "$KEY: inference run failed - see $LOG/bench_${KEY}.log"
      continue
    fi

    ROW=$(parse_run "$LOG/bench_${KEY}.log" "$LOG/tegra_${KEY}.log")
    SIZE=$(stat -c%s "$PLAN" 2>/dev/null)
    python3 - "$KEY" "$M" "$P" "$SIZE" "$ROW" >> "$JSONL" <<'PY'
import json, sys
key, model, prec, size, row = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), json.loads(sys.argv[5])
row.update(key=key, model=model, precision=prec, engine_bytes=size)
print(json.dumps(row))
PY
    python3 - "$ROW" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
p = r.get("power_mw", {})
tot = p.get("VIN_SYS_5V0") or p.get("VDD_IN") or (sum(p.values()) if p else None)
qps = r.get("throughput_qps")
e = (tot / qps) if (tot and qps) else None
print("        p50=%.3fms p99=%.3fms  %.1f img/s  rails=%s  %s"
      % (r.get("gpu_median_ms") or 0, r.get("gpu_p99_ms") or 0, qps or 0,
         ",".join(f"{k}:{v/1000:.1f}W" for k, v in sorted(p.items())[:4]) or "none",
         (f"{e:.0f} mJ/img" if e else "")))
PY
  done
done

# ---------------------------------------------------------------- summary
say "building the summary"
python3 - <<'PY' | tee "$OUT/summary.txt"
import json, os
rows = {}
for ln in open("results/thor_bench.jsonl"):
    try: d = json.loads(ln)
    except Exception: continue
    if "model" in d and d.get("build") != "failed":
        rows[(d["model"], d["precision"])] = d
    elif d.get("build") == "failed":
        rows[(d["model"], d["precision"])] = d

order = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
print(f"{'model':9} {'prec':5} {'p50 ms':>8} {'p99 ms':>8} {'img/s':>8} {'power W':>9} {'mJ/img':>8} {'vs FP16':>8}")
print("-" * 72)
for m in order:
    base = rows.get((m, "fp16"), {}).get("gpu_median_ms")
    for p in ("fp32", "fp16", "int8", "fp8"):
        d = rows.get((m, p))
        if not d: continue
        if d.get("build") == "failed":
            print(f"{m:9} {p:5} {'BUILD FAILED':>44}")
            continue
        pw = d.get("power_mw", {})
        tot = pw.get("VIN_SYS_5V0") or pw.get("VDD_IN") or (sum(pw.values()) if pw else 0)
        qps = d.get("throughput_qps") or 0
        e = tot / qps if (tot and qps) else 0
        su = (base / d["gpu_median_ms"]) if (base and d.get("gpu_median_ms")) else 0
        print(f"{m:9} {p:5} {d.get('gpu_median_ms') or 0:8.3f} {d.get('gpu_p99_ms') or 0:8.3f} "
              f"{qps:8.1f} {tot/1000:9.1f} {e:8.0f} {su:7.2f}x")
print()
print("vs FP16 > 1 means the quantized engine is faster than FP16 on this device.")
PY

tar czf thor_results.tar.gz results 2>/dev/null
echo
say "FINISHED"
echo "-----------------------------------------------------------------"
echo " Send back this one file:   $(pwd)/thor_results.tar.gz"
echo " (it contains the numbers, the raw logs and the environment record)"
[ -s "$OUT/problems.txt" ] && { echo; echo " NOTE - some steps reported problems:"; sed 's/^/   /' "$OUT/problems.txt"; }
echo "-----------------------------------------------------------------"
