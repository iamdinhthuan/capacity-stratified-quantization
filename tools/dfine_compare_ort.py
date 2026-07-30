"""Bisect: run the D-FINE ONNX under ONNX Runtime CPU and diff vs torch npz.

If ORT matches torch -> the ONNX is faithful and the divergence is TensorRT's.
If ORT also diverges -> the torch->ONNX export itself is wrong.

Usage (venv_pilot):
    python compare_ort_vs_ref.py --onnx onnx/dfine_s.onnx --ref ref_torch_s.npz
"""
import argparse

import numpy as np
import onnxruntime as ort

from dfine_infer_trt import preprocess

IMAGES = [
    "/data_nvme/paper/data/coco/images/val2017/000000000139.jpg",
    "/data_nvme/paper/data/coco/images/val2017/000000000285.jpg",
    "/data_nvme/paper/data/coco/images/val2017/000000000632.jpg",
]


def main(onnx_path, ref_path):
    ref = np.load(ref_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    for img in IMAGES:
        key = img.split("/")[-1].split(".")[0]
        inp, w0, h0 = preprocess(img, 640)
        orig = np.array([[w0, h0]], dtype=np.int64)
        feeds = {in_names[0]: inp, in_names[1]: orig}
        labels, boxes, scores = sess.run(None, feeds)
        r_lab = ref[f"{key}_labels"]
        r_box = ref[f"{key}_boxes"]
        r_sco = ref[f"{key}_scores"]
        lm = (labels.flatten()[:50] == r_lab.flatten()[:50]).mean()
        ds = np.abs(scores.flatten()[:50] - r_sco.flatten()[:50]).max()
        db = np.abs(boxes.reshape(-1, 4)[:50] - r_box.reshape(-1, 4)[:50]).max()
        print(f"{key}: label match {lm:.0%} | max|dscore| {ds:.5f} | max|dbox| {db:.3f}px "
              f"| ort top3 s={scores.flatten()[:3]} l={labels.flatten()[:3]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--ref", required=True)
    args = ap.parse_args()
    main(args.onnx, args.ref)
