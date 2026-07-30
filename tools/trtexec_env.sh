# Source this to put the real trtexec CLI (extracted from NVIDIA's TensorRT
# Enterprise tar, matches the pip tensorrt==11.1.0.106 exactly) on PATH.
#   source tools/trtexec_env.sh
#   trtexec --onnx=... --saveEngine=... --fp8 --stronglyTyped
export TRT_SDK_DIR=/home/thuan/traffic/third_party/TensorRT-11.1.0.106
export PATH="$TRT_SDK_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$TRT_SDK_DIR/lib:$LD_LIBRARY_PATH"
