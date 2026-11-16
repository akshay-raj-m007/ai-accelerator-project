#!/usr/bin/env python
"""SQNR figure (initial FP32 / after quantization / after bit flips) for one experiment, Stage 5 design of the other figures.

  python scripts/plot_sqnr_summary.py --sqnr-dir reports/scratch/jfamp/sqnr --model ResNet-18 --prefix resnet18_cifar10 --out plots_preview/resnet18_cifar10/sqnr
  python scripts/plot_sqnr_summary.py --sqnr-dir reports/mobilenetv2_cifar10/sqnr --model MobileNetV2 --prefix mobilenetv2_cifar10 --out reports/mobilenetv2_cifar10/figures/sqnr

Reads sqnr_summary.csv and sqnr_trials.csv written by scripts/sqnr_report.py (committed files only, no model, no GPU).
"""
import argparse
import csv
from pathlib import Path

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_mobilenetv2_summary import COL, GRID, INK, INK2, RC5, save, title

BASE = [("E4M11SN", "e4m11_sn"), ("E4M11", "e4m11_sn"), ("FP16", "fp16"), ("BF16", "bf16"), ("FP8E5", "fp8_e5m2"), ("FP8N", "fp8_e4m3fn"), ("FP8", "fp8_e4m3fn"), ("INT8", "int8_pt"), ("INT4", "int4_pt")]
NICE = {"e4m11_sn": "E4M11-SN", "fp16": "FP16", "bf16": "BF16", "fp8_e4m3fn": "FP8 E4M3", "fp8_e5m2": "FP8 E5M2", "int8_pt": "INT8", "int4_pt": "INT4"}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fmt_of(aid):
    body = aid[2:] if aid.startswith("U_") else aid
    body = body.split("_")[0]
    for k, v in BASE:
        if body == k:
            return v
    return None


def rowname(r):
    if r["kind"] == "final":
        return f"FINAL assignment ({r['assignment_id']})"
    aid = r["assignment_id"][2:]
    parts = aid.split("_")
    f = fmt_of(r["assignment_id"])
    name = NICE[f]
    return name + (f" + {parts[1]}" if len(parts) > 1 else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sqnr-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = Path(a.sqnr_dir)
    S = [r for r in csv.DictReader(open(sd / "sqnr_summary.csv", encoding="utf-8")) if r["assignment_id"] != "FP32"]
    T = list(csv.DictReader(open(sd / "sqnr_trials.csv", encoding="utf-8")))
    n = len(S)
    y = np.arange(n)
    names = [rowname(r) for r in S]
    colr = [INK if r["kind"] == "final" else COL[fmt_of(r["assignment_id"])] for r in S]
    clean = np.array([fnum(r["sqnr_clean_db"]) for r in S])
    med = np.array([fnum(r["weights_k8__sqnr_median_db"]) for r in S])
    p10 = np.array([fnum(r["weights_k8__sqnr_p10_db"]) for r in S])
    worst = np.array([fnum(r["weights_k8__sqnr_worst_finite_db"]) for r in S])
    plt.rcParams.update(RC5)
    fig = plt.figure(figsize=(15, 0.42 * n + 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.0], wspace=0.5, top=1 - 1.6 / (0.42 * n + 5.2), bottom=0.2 if n < 10 else 0.14, left=0.17, right=0.975)
    title(fig, f"SQNR of the stored weights  ({a.model})",
          "Initial trained model (FP32) = reference, error 0, SQNR = +inf.  After quantization = clean SQNR of the stored weights vs the FP32 weights.  After bit flips = same random flips as the accuracy trials.\n"
          "SQNR = 10 log10( sum w^2 / sum (w - stored w)^2 ) over all Conv/Linear weights.  Higher = closer to FP32.", y=0.985)
    ax = fig.add_subplot(gs[0])
    LO = -60
    for i in range(n):
        ax.plot([max(worst[i], LO), p10[i] if p10[i] > LO else LO], [y[i], y[i]], color=colr[i], lw=2, alpha=0.35, solid_capstyle="round")
        ax.scatter(clean[i], y[i], s=70, marker="o", color=colr[i], zorder=3)
        ax.scatter(med[i], y[i], s=70, marker="D", facecolor="white", edgecolor=colr[i], lw=1.6, zorder=3)
        if worst[i] < LO:
            ax.scatter(LO, y[i], s=60, marker="<", color=colr[i], zorder=3)
            ax.text(LO + 2, y[i] - 0.33, f"worst {worst[i]:.0f}", fontsize=7.5, color=INK2, va="center")
        else:
            ax.scatter(worst[i], y[i], s=60, marker="x", color=colr[i], zorder=3)
    ax.set_xlim(LO - 3, 85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("SQNR (dB)")
    ax.set_title("a  8 random weight-bit flips (100 trials)", fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=INK2, markeredgecolor=INK2, markersize=8, label="after quantization (no flips)"),
                       Line2D([], [], marker="D", color="none", markerfacecolor="white", markeredgecolor=INK2, markersize=8, label="after 8 flips, median"),
                       Line2D([], [], marker="x", color=INK2, ls="none", markersize=8, label="after 8 flips, worst trial"),
                       Line2D([], [], color=INK2, alpha=0.35, lw=2, label="worst trial to 10th percentile")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=8.5)
    # b: scale-word flips (INT rows)
    ax2 = fig.add_subplot(gs[1])
    idx = [i for i, r in enumerate(S) if r["scales_k1__n_trials"] not in ("", None)]
    yy = np.arange(len(idx))
    for j, i in enumerate(idx):
        w = fnum(S[i]["scales_k1__sqnr_worst_finite_db"])
        ax2.plot([w, clean[i]], [j, j], color=colr[i], lw=2, alpha=0.35)
        ax2.scatter(clean[i], j, s=70, marker="o", color=colr[i], zorder=3)
        ax2.scatter(w, j, s=60, marker="x", color=colr[i], zorder=3)
        ax2.text(1.02, j, f"worst {w:.0f} dB", fontsize=8.5, color=INK2, va="center", transform=ax2.get_yaxis_transform())
    ax2.set_yticks(yy)
    ax2.set_yticklabels([names[i] for i in idx], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xscale("symlog", linthresh=50)
    ax2.set_xlim(-1200, 60)
    ax2.set_xticks([-1000, -100, 0, 30])
    ax2.set_xticklabels(["-1000", "-100", "0", "30"])
    ax2.minorticks_off()
    ax2.set_xlabel("SQNR (dB, symmetric-log axis)")
    ax2.set_title("b  1 flipped bit in a scale word\n(INT layers only, 100 trials)", fontsize=11)
    ax2.grid(axis="x", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", length=0)
    # c: SQNR vs accuracy drop, weight flips
    ax3 = fig.add_subplot(gs[2])
    for r in S:
        f = fmt_of(r["assignment_id"])
        c = INK if r["kind"] == "final" else COL[f]
        xs = [fnum(t["sqnr_db"]) for t in T if t["assignment_id"] == r["assignment_id"] and t["distribution"] == "weights_k8"]
        ys = [fnum(t["recorded_accuracy_drop_pp"]) for t in T if t["assignment_id"] == r["assignment_id"] and t["distribution"] == "weights_k8"]
        ax3.scatter(xs, ys, s=16, color=c, alpha=0.55, linewidths=0)
    ax3.set_xlim(LO - 3, 85)
    ax3.set_xlabel("SQNR after the flips (dB)")
    ax3.set_ylabel("accuracy drop of the same trial (pp)")
    ax3.set_title("c  SQNR vs accuracy loss\n(one dot = one weight-flip trial)", fontsize=11)
    ax3.grid(color=GRID, lw=0.8)
    ax3.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=COL[f], markeredgecolor="none", markersize=8, label=NICE[f]) for f in dict.fromkeys(fmt_of(r["assignment_id"]) for r in S if r["kind"] != "final")]
    handles.append(Line2D([], [], marker="o", color="none", markerfacecolor=INK, markeredgecolor="none", markersize=8, label="final assignment"))
    ax3.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.09), ncol=2, frameon=False, fontsize=8.5)
    save(fig, Path(a.out) / f"{a.prefix}_sqnr_before_after_faults.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
