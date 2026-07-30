#!/bin/bash
# Build a TensorRT engine with the real trtexec CLI (README §7.3 step 3,
# now literally runnable — see tools/trtexec_env.sh). Cross-check /
# alternative to tools/build_engine.py's Python-API build; useful if you
# also want ModelOpt's --autotune (which needs trtexec, not just the
# Python bindings) or a benchmark summary straight from trtexec's stdout.
#
# Usage: source tools/trtexec_env.sh && tools/build_engine_trtexec.sh <onnx> <engine> <fp8|nvfp4>
set -euo pipefail

ONNX=$1
ENGINE=$2
PRECISION=$3

case "$PRECISION" in
  fp8)       FLAG="--fp8" ;;
  nvfp4|fp4) FLAG="--fp4" ;;
  *) echo "unknown precision: $PRECISION (expected fp8|nvfp4)"; exit 1 ;;
esac

command -v trtexec >/dev/null || { echo "trtexec not on PATH — run: source tools/trtexec_env.sh"; exit 1; }

trtexec --onnx="$ONNX" --saveEngine="$ENGINE" $FLAG \
        --memPoolSize=workspace:8192 --stronglyTyped
