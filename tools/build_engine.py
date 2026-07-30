"""Build a TensorRT engine from an explicit-Q/DQ ONNX (README §7.3 step 3).

Uses the TensorRT Python API directly instead of shelling out to `trtexec`:
this server's `pip install tensorrt` only ships the Python bindings + libs
(confirmed: libnvinfer_builder_resource_sm120.so IS present, so Blackwell
kernels are covered) — the trtexec CLI binary itself comes from NVIDIA's
separate SDK tar/deb, which needs either root or a large extra download we
don't need here.

STRONGLY_TYPED network creation means precision comes entirely from the
Q/DQ + cast nodes already baked into the ONNX by modelopt — no separate
builder precision flags needed, matching modelopt's own recommendation for
deploying explicitly-quantized graphs.

Usage:
    python tools/build_engine.py --onnx exports/yolo26n_fp8.onnx --engine exports/yolo26n_fp8.plan
"""
import argparse

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build(onnx_path, engine_path, workspace_mb):
    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise SystemExit("Engine build failed (see TensorRT log above)")

    engine_bytes = bytes(engine_bytes)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"-> {engine_path} ({len(engine_bytes) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--workspace-mb", type=int, default=8192)
    args = ap.parse_args()
    build(args.onnx, args.engine, args.workspace_mb)
