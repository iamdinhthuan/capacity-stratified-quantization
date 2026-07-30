#!/usr/bin/env bash
# Build the self-contained folder that gets carried to the Jetson AGX Thor.
# Run this on the 5090 box (where the quantized graphs live):
#     bash tools/thor_make_bundle.sh
# It produces  ~/thor_bundle.tar.gz  -- copy that onto a USB stick.
set -uo pipefail
cd ~/coco_journal

MODELS=${MODELS:-"yolo11n yolo11s yolo11m yolo11l yolo11x"}
B=~/thor_bundle
rm -rf "$B"; mkdir -p "$B/onnx"

echo "collecting graphs..."
for M in $MODELS; do
  for f in "$M.onnx" "${M}_fp16.onnx" "${M}_int8.onnx" "${M}_int8_hp32.onnx" "${M}_fp8.onnx"; do
    [ -f "exports/coco_pilot/$f" ] && cp "exports/coco_pilot/$f" "$B/onnx/" && echo "  + $f"
  done
done
cp tools/thor_bench.sh "$B/"
cp tools/THOR_README.txt "$B/" 2>/dev/null || true

# record exactly which graphs travelled, so the device results can be tied
# back to the same quantize-once artifacts used on the desktop and on Orin
( cd "$B/onnx" && md5sum *.onnx > ../onnx_checksums.txt )

du -sh "$B"
tar czf ~/thor_bundle.tar.gz -C ~ thor_bundle
echo
echo "-> ~/thor_bundle.tar.gz  ($(du -h ~/thor_bundle.tar.gz | cut -f1))"
echo "   copy it to a USB stick, unpack on the Thor, and run:  bash thor_bench.sh"
