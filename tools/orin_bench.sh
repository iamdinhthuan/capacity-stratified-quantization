#!/usr/bin/env bash
# Latency + power benchmark for TensorRT engines ON a Jetson (runs on-device).
# For each engines/*.plan: trtexec timed run (GPU compute p50/p95/p99) with a
# parallel tegrastats log; VDD_IN average over the steady-state window.
# Output: one JSON line per engine into bench_results.jsonl
#
# Usage (on the Jetson):  bash orin_bench.sh [engines_dir]
set -u
DIR=${1:-~/qtsd_edge/engines}
OUT=~/qtsd_edge/bench_results.jsonl
LOG=~/qtsd_edge/logs
T=/usr/src/tensorrt/bin/trtexec
mkdir -p "$LOG"
: > "$OUT"

for E in "$DIR"/*.plan; do
  NAME=$(basename "$E" .plan)
  echo "=== bench $NAME ==="
  # power sampling in the background (200 ms interval)
  tegrastats --interval 200 > "$LOG/tegra_${NAME}.log" &
  TPID=$!
  sleep 2   # idle baseline present at head of log
  "$T" --loadEngine="$E" --warmUp=1000 --duration=30 --useSpinWait \
      > "$LOG/bench_${NAME}.log" 2>&1
  RC=$?
  sleep 1
  kill $TPID 2>/dev/null; wait $TPID 2>/dev/null
  if [ $RC -ne 0 ]; then
    echo "{\"engine\":\"$NAME\",\"error\":\"trtexec_failed\"}" >> "$OUT"
    tail -3 "$LOG/bench_${NAME}.log"; continue
  fi
  # detailed stats line (NOT the trailing "Total GPU Compute Time" line)
  STATS=$(grep "GPU Compute Time: min" "$LOG/bench_${NAME}.log" | tail -1)
  QPS=$(grep -oE "Throughput: [0-9.]+" "$LOG/bench_${NAME}.log" | tail -1 | grep -oE "[0-9.]+")
  MEAN=$(echo "$STATS"   | grep -oE "mean = [0-9.]+"      | grep -oE "[0-9.]+")
  MEDIAN=$(echo "$STATS" | grep -oE "median = [0-9.]+"    | grep -oE "[0-9.]+")
  P99=$(echo "$STATS"    | grep -oE "percentile\(99%\) = [0-9.]+" | grep -oE "[0-9.]+" | tail -1)
  # VDD_IN average over the run (skip the 2 s idle head), Orin Nano format: "VDD_IN 4970mW/4900mW"
  PW=$(awk 'NR>10 {for(i=1;i<=NF;i++) if($i=="VDD_IN"){split($(i+1),a,"/"); gsub("mW","",a[1]); s+=a[1]; n++}} END{if(n>0) printf "%.0f", s/n; else print "-1"}' "$LOG/tegra_${NAME}.log")
  echo "{\"engine\":\"$NAME\",\"mean_ms\":${MEAN:-null},\"median_ms\":${MEDIAN:-null},\"p99_ms\":${P99:-null},\"qps\":${QPS:-null},\"vdd_in_mw\":${PW:-null}}" >> "$OUT"
  echo "  mean=${MEAN}ms median=${MEDIAN}ms p99=${P99}ms qps=${QPS} VDD_IN=${PW}mW"
done
echo "-> $OUT"
