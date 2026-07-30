"""Export a D-FINE COCO checkpoint to deployment ONNX (static batch 1, 640x640).

Mirrors D-FINE/tools/deployment/export_onnx.py exactly (deploy-mode model +
embedded DFINEPostProcessor: sigmoid + flat topk-300, labels 0..79 contiguous,
boxes cxcywh->xyxy * orig_target_sizes[w,h] => absolute original-image pixels)
with two deliberate changes for the TRT parity pipeline:
  - static shapes: images (1,3,640,640), orig_target_sizes (1,2) int64 —
    no dynamic axes, so the strongly-typed TRT engine is fully static;
  - opset selectable (default 16, the stock choice; issue #153 reports
    16-vs-17 GridSample differences under TRT FP16).

Run with /data_nvme/paper/.venv_dfine/bin/python (torch 2.6 + faster-coco-eval).

Usage:
    python export_dfine_onnx.py --config D-FINE/configs/dfine/dfine_hgnetv2_n_coco.yml \
        --resume ckpt/dfine_n_coco.pth --out onnx/dfine_n.onnx [--opset 16] [--no-simplify]
"""
import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "D-FINE"))

import torch
import torch.nn as nn

from src.core import YAMLConfig


# ---- explicit gather-bilinear deformable core (dfine-cpp spike, verified
# equivalent to F.grid_sample align_corners=False + zeros padding) ----------
def bilinear_gather(value_l, grid_l, h, w):
    M, c = value_l.shape[0], value_l.shape[1]
    Lq, P = grid_l.shape[1], grid_l.shape[2]
    gx, gy = grid_l[..., 0], grid_l[..., 1]
    ix = (gx + 1) * w / 2 - 0.5
    iy = (gy + 1) * h / 2 - 0.5
    x0 = torch.floor(ix)
    y0 = torch.floor(iy)
    x1 = x0 + 1
    y1 = y0 + 1
    wx1 = ix - x0
    wx0 = 1 - wx1
    wy1 = iy - y0
    wy0 = 1 - wy1
    vflat = value_l.reshape(M, c, h * w)

    def corner(xc, yc, wgt):
        valid = ((xc >= 0) & (xc <= w - 1) & (yc >= 0) & (yc <= h - 1)).to(value_l.dtype)
        xcl = xc.clamp(0, w - 1)
        ycl = yc.clamp(0, h - 1)
        idx = (ycl * w + xcl).long().reshape(M, 1, Lq * P).expand(M, c, Lq * P)
        g = torch.gather(vflat, 2, idx).reshape(M, c, Lq, P)
        return g * (wgt * valid).unsqueeze(1)

    return (corner(x0, y0, wx0 * wy0) + corner(x1, y0, wx1 * wy0)
            + corner(x0, y1, wx0 * wy1) + corner(x1, y1, wx1 * wy1))


def explicit_deformable_core(value, value_spatial_shapes, sampling_locations,
                             attention_weights, num_points_list, method="default"):
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape
    grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
    grids_list = grids.split(num_points_list, dim=-2)
    sampled = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, int(h), int(w))
        sampled.append(bilinear_gather(value_l, grids_list[level], int(h), int(w)))
    attn = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    out = (torch.concat(sampled, dim=-1) * attn).sum(-1).reshape(bs, n_head * c, Len_q)
    return out.permute(0, 2, 1)


def patch_explicit_deform(model):
    n = 0
    for layer in model.decoder.decoder.layers:
        layer.cross_attn.ms_deformable_attn_core = functools.partial(
            explicit_deformable_core, method="default")
        n += 1
    print(f"patched {n} cross_attn cores -> explicit gather-bilinear")


def main(config, resume, out, opset, simplify, imgsz, explicit_deform=False):
    cfg = YAMLConfig(config, resume=resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)
    print(f"loaded {'ema' if 'ema' in checkpoint else 'model'} state from {resume}")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = Model().eval()

    data = torch.rand(1, 3, imgsz, imgsz)
    size = torch.tensor([[imgsz, imgsz]], dtype=torch.int64)
    with torch.no_grad():
        labels, boxes, scores = model(data, size)
    print(f"sanity fwd: labels {tuple(labels.shape)} {labels.dtype}, "
          f"boxes {tuple(boxes.shape)} {boxes.dtype}, scores {tuple(scores.shape)} {scores.dtype}")

    if explicit_deform:
        patch_explicit_deform(model.model)
        with torch.no_grad():
            l2, b2, s2 = model(data, size)
        print(f"explicit-deform torch parity: max|dscore|={float((scores - s2).abs().max()):.3e} "
              f"max|dbox|={float((boxes - b2).abs().max()):.3e}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        (data, size),
        out,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=None,          # static batch 1
        opset_version=opset,
        verbose=False,
        do_constant_folding=True,
    )

    import onnx
    m = onnx.load(out)
    onnx.checker.check_model(m)
    if simplify:
        import onnxsim
        m_sim, ok = onnxsim.simplify(
            out, test_input_shapes={"images": list(data.shape),
                                    "orig_target_sizes": list(size.shape)})
        onnx.save(m_sim, out)
        print(f"onnxsim: {ok}")
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=16)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--explicit-deform", action="store_true",
                    help="replace grid_sample deformable core with gather-bilinear (TRT-safe)")
    args = ap.parse_args()
    main(args.config, args.resume, args.out, args.opset, not args.no_simplify, args.imgsz,
         args.explicit_deform)
