"""Aggregate metrics/{model}_{precision}.json into README §9 Bảng 1-4.

Every number is read straight from a metrics json — nothing here is decided
by eyeballing a run; re-running this after adding a new precision/model just
means re-running this one command (README §8: "chạy lại một lệnh là ra lại
mọi bảng/hình trong bài").

Usage:
    python tools/make_tables.py --metrics-dir metrics --out-dir tables
"""
import argparse
import glob
import json
import math
import os

import pandas as pd

from common import REPO_ROOT, load_size_bins

PRECISION_ORDER = ["fp32", "fp16", "int8_ptq", "int8_qat", "fp8", "fp4"]


def load_all(metrics_dir):
    rows = {}
    for path in glob.glob(os.path.join(metrics_dir, "*.json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if "model" not in d or "precision" not in d:
            continue  # not an eval_stratified output (e.g. gt_test.json)
        rows[(d["model"], d["precision"])] = d
    return rows


def table1_overview(rows):
    out = []
    for (model, prec), d in rows.items():
        sysm = d.get("system", {})
        out.append({
            "Detector": model, "Precision": prec,
            "mAP50-95": d["overall"]["mAP50-95"], "mAP50": d["overall"]["mAP50"],
            "mAP-small": d["coco_bin_ap"]["small"]["mAP50-95"],
            "FPS(bs=1)": sysm.get("fps_bs1"), "Engine(MB)": sysm.get("engine_size_mb"),
            # measure_system.py writes power_w_mean; accept the older power_w too
            "Power(W)": sysm.get("power_w_mean", sysm.get("power_w")),
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    order = {p: i for i, p in enumerate(PRECISION_ORDER)}
    df["_o"] = df["Precision"].map(lambda p: order.get(p, 99))
    return df.sort_values(["Detector", "_o"]).drop(columns="_o")


def table2_size_degradation(rows, metric="mAP50-95"):
    bins = [b["name"] for b in load_size_bins()["height_bins"]]
    out = []
    for model in {m for m, _ in rows}:
        base = rows.get((model, "fp32"))
        if base is None:
            continue
        for prec in PRECISION_ORDER:
            if prec == "fp32" or (model, prec) not in rows:
                continue
            d = rows[(model, prec)]
            row = {"Detector": model, "Precision": prec}
            for b in bins:
                delta = base["height_bin_ap"][b][metric] - d["height_bin_ap"][b][metric]
                row[f"Delta_{b}"] = delta
            xs, xl = row.get(f"Delta_{bins[0]}"), row.get(f"Delta_{bins[-1]}")
            row["SUR(Δxs/Δxl)"] = (xs / xl) if xl not in (0, None) and not math.isnan(xl) else float("nan")
            out.append(row)
    return pd.DataFrame(out)


def table3_superclass_degradation(rows, metric="mAP50-95"):
    out = []
    for model in {m for m, _ in rows}:
        base = rows.get((model, "fp32"))
        if base is None:
            continue
        scs = list(base["superclass_ap"].keys())
        for prec in PRECISION_ORDER:
            if prec == "fp32" or (model, prec) not in rows:
                continue
            d = rows[(model, prec)]
            row = {"Detector": model, "Precision": prec}
            for sc in scs:
                row[f"Delta_{sc}"] = base["superclass_ap"][sc][metric] - d["superclass_ap"][sc][metric]
            out.append(row)
    return pd.DataFrame(out)


def table4_ewd(rows, pinhole_cfg):
    h_real, f_px = pinhole_cfg["H_real_m"], pinhole_cfg["f_px"]
    out = []
    for model in {m for m, _ in rows}:
        base = rows.get((model, "fp32"))
        s_fp32 = base["ewd"]["s_star_px"] if base else None
        for prec in PRECISION_ORDER:
            if (model, prec) not in rows:
                continue
            d = rows[(model, prec)]
            s_q = d["ewd"]["s_star_px"]
            row = {"Detector": model, "Precision": prec, "EWD(px)": s_q}
            if prec != "fp32" and s_fp32 is not None and s_q is not None:
                row["DeltaEWD(px)"] = s_q - s_fp32
                if s_fp32 > 0 and s_q > 0:
                    row["Delta_distance(m)_illustrative"] = f_px * h_real * (1.0 / s_fp32 - 1.0 / s_q)
            out.append(row)
    return pd.DataFrame(out)


def main(metrics_dir, out_dir):
    rows = load_all(metrics_dir)
    if not rows:
        raise SystemExit(f"No eval_stratified outputs found in {metrics_dir}")
    os.makedirs(out_dir, exist_ok=True)
    size_bins = load_size_bins()

    tables = {
        "table1_overview": table1_overview(rows),
        "table2_size_degradation": table2_size_degradation(rows),
        "table3_superclass_degradation": table3_superclass_degradation(rows),
        "table4_ewd": table4_ewd(rows, size_bins["ewd"]["pinhole"]),
    }
    for name, df in tables.items():
        csv_path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))
        print(f"-> {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default=os.path.join(REPO_ROOT, "metrics"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "tables"))
    args = ap.parse_args()
    main(args.metrics_dir, args.out_dir)
