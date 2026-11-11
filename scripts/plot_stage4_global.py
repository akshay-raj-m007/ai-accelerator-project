"""Plots of the Stage 4 global screening (runbook 6.1: random encoded weights across all layers, k = 1 and 8).

Reads only fault_trials.jsonl (raw records); every dot is one trial, the bar is the mean of 20 trials, the tick is the p90.
Usage: python scripts/plot_stage4_global.py [--dir reports/scratch/jfamp/fault_screening] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FORMATS = ["e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt"]
LABEL = {"e4m11_sn": "E4M11-SN", "fp16": "FP16", "fp8_e4m3fn": "FP8 E4M3FN", "int8_pt": "INT8", "int4_pt": "INT4"}
COLOR = {"e4m11_sn": "#1b9e77", "fp16": "#d95f02", "fp8_e4m3fn": "#7570b3", "int8_pt": "#e7298a", "int4_pt": "#66a61e"}
# x slots: the INT "value" group sits in the "fraction" slot (same role: payload bits); INT has no exponent group
SLOTS = [("sign", "sign"), ("exponent", "exponent"), ("fraction", "fraction / value"), ("all", "all bits")]


def group_slot(fmt, group):
    return "fraction" if group == "value" else group


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["layer_name"] not in ("GLOBAL", "GLOBAL-SCALES"):
                continue
            key = (r["scope_kind"], r["format"], group_slot(r["format"], r["fault_group"]), r["requested_k"])
            out.setdefault(key, []).append(r["clean_accuracy"] - r["fault_accuracy"])
    return out


def panel(ax, data, scope, k, formats, title):
    present = [f for f in formats if any((scope, f, s, k) in data for s, _ in SLOTS)]
    w = 0.8 / max(1, len(present))
    rng = np.random.default_rng(0)
    for i, fmt in enumerate(present):
        for j, (slot, _) in enumerate(SLOTS):
            v = data.get((scope, fmt, slot, k))
            if v is None:
                continue
            x = j - 0.4 + w * (i + 0.5)
            ax.bar(x, np.mean(v), width=w * 0.92, color=COLOR[fmt], alpha=0.55, label=LABEL[fmt] if j == 0 or (scope, fmt, SLOTS[0][0], k) not in data else None)
            ax.scatter(x + rng.uniform(-w * 0.3, w * 0.3, len(v)), v, s=7, color="k", alpha=0.6, zorder=3)
            ax.plot([x - w * 0.46, x + w * 0.46], [np.percentile(v, 90)] * 2, color="k", lw=1.6, zorder=4)
    ax.set_xticks(range(len(SLOTS)))
    ax.set_xticklabels([n for _, n in SLOTS])
    ax.set_ylabel("accuracy drop (pp) = clean - faulted")
    ax.set_title(title)
    ax.axhline(0, color="grey", lw=0.6)
    ax.grid(axis="y", alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for a, b in zip(h, l):
        seen.setdefault(b, a)
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/scratch/jfamp/fault_screening")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    out = Path(a.out) if a.out else d / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = load(d / "fault_trials.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, k in zip(axes, (1, 8)):
        panel(ax, data, "weights", k, FORMATS, f"Global screening, k = {k} flipped bit(s) in 11.16 M encoded weights")
    fig.suptitle("Stage 4 global screening (weights): bar = mean of 20 trials, dots = individual trials, black tick = p90; "
                 "1,000 validation images", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "global_screening_weights.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, k in zip(axes, (1, 8)):
        panel(ax, data, "scales", k, ["int8_pt", "int4_pt"], f"Global scale-word faults, k = {k} (21 float32 scales)")
    fig.suptitle("Stage 4 extension: INT per-tensor scale words (bar = mean of 20 trials, dots = trials, tick = p90)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "global_screening_int_scales.png", dpi=160)
    plt.close(fig)

    # zoomed view of the weight payload for the formats whose mean is small (everything except FP16)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, k in zip(axes, (1, 8)):
        panel(ax, data, "weights", k, [f for f in FORMATS if f != "fp16"], f"Zoom without FP16, k = {k}")
    fig.tight_layout()
    fig.savefig(out / "global_screening_weights_zoom_no_fp16.png", dpi=160)
    plt.close(fig)
    print("written:", *sorted(p.name for p in out.glob("global_screening*.png")))


if __name__ == "__main__":
    main()
