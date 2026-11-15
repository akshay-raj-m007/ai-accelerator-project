#!/usr/bin/env python
"""MobileNetV2 / CIFAR-10 figures in the design of the ResNet-18 figures (plot_stage4_summary.py and plot_stage5_summary.py).

Runs on committed CSV / JSON files only (no model, no data, no GPU):

  python scripts/plot_mobilenetv2_summary.py --reports reports/mobilenetv2_cifar10 --out reports/mobilenetv2_cifar10/figures

Design rules copied from the ResNet-18 scripts
  * stage4/  = Stage 4 look: white background, black-edged bars, per-format colours of plot_stage4_summary.py, 'Reds' heatmaps with the value printed in every cell.
  * stage6/  = Stage 5 look: off-white surface, left-aligned bold title with grey sub-title, per-format colours of plot_stage5_summary.py, blue (Q, R) and orange (scale word)
               sequential heatmaps with the value in every cell, assignment grid with one coloured cell per layer.
  The two formats that ResNet-18 did not have (BF16, FP8 E5M2) take the next free colour of the same palette; the 7 formats keep the ResNet order (E4M11-SN, FP16, ..., INT8, INT4).
Differences that follow from the data, not from taste
  * Stage 4 of MobileNetV2 followed the protocol (sign / exponent / fraction groups, 30 global and 10 layer-wise trials); it has no 'any bit' group, so figures 2, 4 and 5 use the groups
    (figure 5 pools them bit-weighted, as the optimiser does).
  * The ResNet-18 figures of weight / activation distributions, activation growth and feature maps need per-trial tensors that the MobileNetV2 runs did not save: not reproduced.
"""
import argparse
import csv
import json
from pathlib import Path

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from scipy.stats import spearmanr

from fr import stage3 as S3

FORMATS = ["e4m11_sn", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "int8_pt", "int4_pt"]
BITS = {"e4m11_sn": 16, "fp16": 16, "bf16": 16, "fp8_e4m3fn": 8, "fp8_e5m2": 8, "int8_pt": 8, "int4_pt": 4}
NAME4 = {"e4m11_sn": "E4M11-SN", "fp16": "FP16", "bf16": "BF16", "fp8_e4m3fn": "FP8 (E4M3)", "fp8_e5m2": "FP8 (E5M2)", "int8_pt": "INT8", "int4_pt": "INT4"}
LABEL = {"e4m11_sn": "E4M11-SN", "fp16": "FP16", "bf16": "BF16", "fp8_e4m3fn": "FP8 E4M3FN", "fp8_e5m2": "FP8 E5M2", "int8_pt": "INT8", "int4_pt": "INT4"}
SHORTF = {"e4m11_sn": "E4M11", "fp16": "FP16", "bf16": "BF16", "fp8_e4m3fn": "FP8", "fp8_e5m2": "FP8-E5", "int8_pt": "INT8", "int4_pt": "INT4"}
# Stage 4 palette (plot_stage4_summary.py) + two extra
FCOLOR = {"e4m11_sn": "#2a9d8f", "fp16": "#e76f51", "fp8_e4m3fn": "#8d6cab", "int8_pt": "#3d7ab8", "int4_pt": "#9aa5b1", "bf16": "#e9c46a", "fp8_e5m2": "#b56576"}
GCOLOR = {"sign": "#4c78a8", "exponent": "#e4572e", "fraction": "#59a14f", "all": "#8c8c8c"}
# Stage 5 palette (plot_stage5_summary.py) + two extra slots of the same reference palette
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
COL = {"e4m11_sn": "#2a78d6", "fp16": "#eb6834", "fp8_e4m3fn": "#1baf7a", "int8_pt": "#6250d6", "int4_pt": "#e34948", "bf16": "#eda100", "fp8_e5m2": "#e87ba4"}
SEQ_BLUE = LinearSegmentedColormap.from_list("seqblue", ["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("seqorange", ["#fef3ec", "#f6c3a8", "#eb6834", "#b0451a", "#6e2608"])
TOK = {"FP16": "fp16", "BF16": "bf16", "E4M11": "e4m11_sn", "FP8N": "fp8_e4m3fn", "FP8E5": "fp8_e5m2", "INT8": "int8_pt", "INT4": "int4_pt"}
RC4 = {"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
       "axes.spines.top": False, "axes.spines.right": False, "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black"}
RC5 = {"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
       "axes.edgecolor": GRID, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
       "font.family": "DejaVu Sans"}


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def f(x, d=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def short(n):
    """features.N.conv.* -> bN.exp / bN.dw / bN.proj (block 1 has no expansion), stem, head, fc."""
    if n == "features.0.0":
        return "stem"
    if n == "features.18.0":
        return "head"
    if n == "classifier.1":
        return "fc"
    p = n.split(".")
    b = int(p[1])
    tail = ".".join(p[3:])
    if b == 1:
        return f"b{b}." + {"0.0": "dw", "1": "proj"}[tail]
    return f"b{b}." + {"0.0": "exp", "1.0": "dw", "2": "proj"}[tail]


def title(fig, main, sub, y=0.985):
    fig.text(0.012, y, main, fontsize=15, fontweight="bold", va="top", color=INK)
    fig.text(0.012, y - 0.034, sub, fontsize=10.5, va="top", color=INK2, linespacing=1.35)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  ", path.name)


class D:
    def __init__(self, rep):
        r = Path(rep)
        self.r = r
        self.clean = {x["format_key"]: x for x in rd(r / "jfamp/quantization/clean_accuracy.csv")}
        self.qm = rd(r / "jfamp/quantization/quantization_metrics.csv")
        self.stor = {x["format_key"]: x for x in rd(r / "jfamp/quantization/storage_summary.csv")}
        self.gs = rd(r / "stage_04_encoded_fault_injection/global_summary.csv")
        self.ls = rd(r / "stage_04_encoded_fault_injection/layerwise_summary.csv")
        self.cand = rd(r / "stage_06_fisher/jfamp_candidate_table.csv")
        self.fq = rd(r / "stage_06_fisher/fisher_q.csv")
        self.s3 = rd(r / "stage_03_layerwise_sensitivity/layerwise_sensitivity_results.csv")
        self.prot = rd(r / "stage_05_protection/protection_summary.csv")
        self.pstor = rd(r / "stage_05_protection/protection_storage_summary.csv")
        self.scale = rd(r / "stage_06_optimization/scale_faults/scale_summary.csv")
        self.e2e = {x["assignment_id"]: x for x in rd(r / "stage_06_fisher/end_to_end_assignments.csv")}
        self.m = json.load(open(r / "stage_06_fisher/end_to_end_assignment_metrics.json"))
        self.A = {a["assignment_id"]: a for a in self.m["assignments"]}
        self.asg = json.load(open(r / "stage_06_fisher/assignment.json"))
        self.test = json.load(open(r / "stage_06_fisher/final_test_evaluation.json"))
        self.testA = json.load(open(r / "stage_06_optimization/final_test_evaluation.json"))
        self.sweep = rd(r / "stage_06_fisher/normalization_lambda_sweep.csv")
        self.trials = [json.loads(l) for l in open(r / "stage_06_fisher/final_validation_fault_trials.jsonl", encoding="utf-8")]
        self.layers = [x["layer_name"] for x in self.asg["selected"]]
        self.nw = {x["layer_name"]: int(x["num_weights"]) for x in self.asg["selected"]}
        self.tab = {(x["layer_name"], x["datatype"], x["protection"]): x for x in self.cand}
        self.sq = {x["format_key"]: f(x["sqnr_db"]) for x in self.qm if x["module_name"] == "GLOBAL"}
        self.s3m = {(x["layer_name"], x["format_key"]): f(x["loss_change"]) for x in self.s3}


# ======================================================================================== Stage 4 look (white, black-edged, FCOLOR)
def s4_overview(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    x = np.arange(len(FORMATS))
    ax = axes[0]
    ref = f(d.clean["fp32"]["val_accuracy_percent"])
    vals = [f(d.clean[k]["val_accuracy_percent"]) for k in FORMATS]
    ax.axhline(ref, color="k", ls="--", lw=1, label=f"original FP32 model ({ref:.2f} %)")
    ax.scatter(x, vals, s=140, c=[FCOLOR[k] for k in FORMATS], zorder=3, edgecolors="k")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f} %", (i, v), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=9, bbox=dict(fc="white", ec="none", pad=1))
    ax.set_ylim(min(vals) - 1, ref + 1.6)
    ax.set_title("1. Accuracy after compressing the weights\n(5,000 validation images, higher is better)", fontsize=10)
    ax.set_ylabel(f"accuracy (%)  - note: axis starts at {min(vals) - 1:.0f}")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    s = [d.sq[k] for k in FORMATS]
    ax.bar(x, s, color=[FCOLOR[k] for k in FORMATS], edgecolor="k")
    for i, v in enumerate(s):
        ax.text(i, v + 1, f"{v:.1f} dB", ha="center", fontsize=9)
    ax.set_ylim(0, max(s) * 1.15)
    ax.set_title("2. How close are the compressed weights to the original?\n(SQNR in dB, higher = closer)", fontsize=10)
    ax.set_ylabel("SQNR (dB)")
    ax = axes[2]
    b = [f(d.stor[k]["actual_bits_per_weight"]) for k in FORMATS]
    ax.bar(x, b, color=[FCOLOR[k] for k in FORMATS], edgecolor="k")
    for i, k in enumerate(FORMATS):
        ax.text(i, b[i] + 0.4, f"{b[i]:.0f} bits\n({f(d.stor[k]['compression_ratio_weights']):.0f}x smaller)", ha="center", fontsize=9)
    ax.axhline(32, color="k", ls="--", lw=1)
    ax.text(len(FORMATS) - 0.5, 32.5, "original FP32 = 32 bits", ha="right", fontsize=8)
    ax.set_ylim(0, 37)
    ax.set_title("3. Memory used per weight\n(lower = smaller model)", fontsize=10)
    ax.set_ylabel("bits per weight")
    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels([NAME4[k] for k in FORMATS], rotation=20)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("What each number format costs and how good it is, before any fault (MobileNetV2, 53 layers)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out / "mobilenetv2_cifar10_1_format_overview.png")


GROUPS = [("sign", "sign bit"), ("exponent", "exponent bits"), ("fraction", "fraction / value bits")]


def grouped_bars(ax, rows, k):
    slot_w = 0.24
    for i, fm in enumerate(FORMATS):
        cleans = []
        for j, (g, _) in enumerate(GROUPS):
            r = [r for r in rows if r["format"] == fm and int(r["k"]) == k and (r["fault_group"] == g or (g == "fraction" and r["fault_group"] == "value"))]
            x = i + (j - 1) * slot_w * 1.08
            if not r:
                ax.text(x, 3, "n/a", rotation=90, ha="center", va="bottom", fontsize=7, color="grey")
                continue
            r = r[0]
            cl, fa, worst = f(r["clean_accuracy"]), f(r["fault_accuracy_mean"]), f(r["clean_accuracy"]) - f(r["drop_pp_worst"])
            cleans.append(cl)
            ax.bar(x, fa, width=slot_w, color=GCOLOR[g], edgecolor="k", lw=0.5, label=dict(GROUPS)[g] if i == 0 else None)
            ax.plot([x, x], [fa, worst], color="k", lw=1)
            ax.plot([x - slot_w * 0.3, x + slot_w * 0.3], [worst, worst], color="k", lw=1.6)
            dd = cl - fa
            if dd >= 1:
                ax.text(x, fa + 1.5, f"-{dd:.0f}" if dd >= 10 else f"-{dd:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        if cleans:
            c = float(np.mean(cleans))
            ax.plot([i - 0.45, i + 0.45], [c, c], color="k", ls="--", lw=1.2, label="accuracy without faults" if i == 0 else None)
    ax.set_xticks(range(len(FORMATS)))
    ax.set_xticklabels([NAME4[k_] for k_ in FORMATS], fontsize=9.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("accuracy after faults (%)  - higher is better")
    ax.grid(axis="y", alpha=0.3)


def s4_global(d, out):
    rows = [r for r in d.gs if r["scope_kind"] == "weights"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), sharey=True)
    for ax, k in zip(axes, (1, 8)):
        grouped_bars(ax, rows, k)
        ax.set_title(f"{k} random bit{'s' if k > 1 else ''} flipped among the 2.2 million stored weights")
    h, l = axes[0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    seen["worst of the 30 trials (black tick)"] = Line2D([], [], color="k", lw=1.6)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=len(seen), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Global screening: accuracy of the model after random bit flips in the stored weights (all 2.2 million weights)", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005, "Each bar = mean accuracy of 30 independent fault trials (1,000 validation images each), flips restricted to one bit group. Number above a bar = mean accuracy lost "
                         "(percentage points). n/a = this bit group does not exist for the format.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    save(fig, out / "mobilenetv2_cifar10_2_global_screening_weights.png")


def s4_scales(d, out):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    slot_w = 0.24
    for i, fm in enumerate(("int8_pt", "int4_pt")):
        rows = [r for r in d.scale if r["format_key"] == fm]
        cl = np.mean([f(r["clean_accuracy"]) for r in rows])
        for j, (g, lab) in enumerate((("sign", "scale: sign bit (1 bit)"), ("exponent", "scale: exponent bits (8 bits)"), ("fraction", "scale: fraction bits (23 bits)"))):
            drop = np.mean([f(r[f"mean_drop_pp_{g}"]) for r in rows])
            x = i + (j - 1) * slot_w * 1.08
            ax.bar(x, cl - drop, width=slot_w, color=GCOLOR[g], edgecolor="k", lw=0.5, label=lab if i == 0 else None)
            if drop >= 1:
                ax.text(x, cl - drop + 1.5, f"-{drop:.0f}" if drop >= 10 else f"-{drop:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.plot([i - 0.45, i + 0.45], [cl, cl], color="k", ls="--", lw=1.2, label="accuracy without faults" if i == 0 else None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["INT8 (per-tensor)", "INT4 (per-tensor)"])
    ax.set_ylim(0, 108)
    ax.set_ylabel("accuracy after faults (%)  - higher is better")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    ax.set_title("1 flipped bit in the float32 scale of an INT layer\n(every one of the 32 scale bits flipped once, in every INT layer: 106 layer/format cases)")
    fig.suptitle("Global screening of the SCALE words (INT8 / INT4): one scale bit can collapse a layer", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out / "mobilenetv2_cifar10_3_global_screening_scales.png")


def s4_heatmaps(d, out):
    layers = d.layers
    panels = [("exponent", 8, [k for k in FORMATS if not k.startswith("int")], "8 flips, exponent bits only\n(floating-point formats only)"),
              ("sign", 8, FORMATS, "8 flips, sign bit only"),
              ("fraction", 8, FORMATS, "8 flips, fraction bits\n(INT: value bits)")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 17), gridspec_kw={"width_ratios": [5, 7, 7]}, sharey=True)
    im = None
    for ax, (g, k, fmts, ttl) in zip(axes, panels):
        M = np.full((len(layers), len(fmts)), np.nan)
        for r in d.ls:
            if int(r["k"]) == k and r["format"] in fmts and (r["fault_group"] == g or (g == "fraction" and r["fault_group"] == "value")):
                M[layers.index(r["layer_name"]), fmts.index(r["format"])] = f(r["drop_pp_mean"])
        im = ax.imshow(M, cmap="Reds", vmin=0, vmax=60, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, "0" if abs(v) < 0.05 else f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v > 35 else "black")
        ax.set_xticks(range(len(fmts)))
        ax.set_xticklabels([NAME4[k_] for k_ in fmts], rotation=30, ha="right")
        ax.set_title(ttl, fontsize=10)
        ax.spines[:].set_visible(False)
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels([f"{short(l)}  ({l})" for l in layers], fontsize=7)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
    cb.set_label("mean accuracy lost (percentage points; colour capped at 60)")
    fig.suptitle("Which layer hurts most when its stored weights are corrupted? (faults injected into ONE layer at a time, mean of 10 trials)\n"
                 "rows = layers from the first (top) to the last (bottom); darker red = more accuracy lost (tiny negative numbers are noise)", fontsize=12, fontweight="bold")
    path = out / "mobilenetv2_cifar10_4_layer_heatmaps.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ", path.name)


def pooled(d, fm, k):
    """Bit-weighted pooled accuracy over the fault groups of the format (stands in for the 'any bit' group that the MobileNetV2 protocol does not have)."""
    cl = fa = ws = 0.0
    for g in ("sign", "exponent", "fraction", "value"):
        r = [r for r in d.gs if r["format"] == fm and r["scope_kind"] == "weights" and r["fault_group"] == g and int(r["k"]) == k]
        if r:
            nb = len(S3.group_bits(fm, g))
            cl += nb * f(r[0]["clean_accuracy"])
            fa += nb * f(r[0]["fault_accuracy_mean"])
            ws += nb
    return cl / ws, fa / ws


def s4_quality(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for ax, k in zip(axes, (1, 8)):
        for fm in FORMATS:
            cl, fa = pooled(d, fm, k)
            ax.annotate("", xy=(d.sq[fm], fa), xytext=(d.sq[fm], cl), arrowprops=dict(arrowstyle="->", color=FCOLOR[fm], lw=2))
            ax.scatter([d.sq[fm]], [cl], s=90, facecolors="white", edgecolors=FCOLOR[fm], lw=2, zorder=3)
            ax.scatter([d.sq[fm]], [fa], s=90, color=FCOLOR[fm], edgecolors="k", zorder=4)
            ax.annotate(f"{NAME4[fm]}\n{cl:.1f} -> {fa:.1f} %", (d.sq[fm], min(cl, fa)), textcoords="offset points", xytext=(0, -36), ha="center", fontsize=8.5)
        ax.set_ylim(60, 96)
        ax.set_xlim(0, max(d.sq[k_] for k_ in FORMATS) + 12)
        ax.set_xlabel("SQNR of the compressed weights (dB): higher = closer to the original")
        ax.set_title(f"{k} random bit{'s' if k > 1 else ''} flipped in the stored weights")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy (%)  - note: axis starts at 60")
    axes[0].scatter([], [], s=90, facecolors="white", edgecolors="k", lw=2, label="hollow = before faults")
    axes[0].scatter([], [], s=90, color="grey", edgecolors="k", label="filled = after faults (mean of 30 trials)")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Compression quality (SQNR) vs. robustness: the arrow is the accuracy lost to random bit flips\n(bit-weighted over the sign / exponent / fraction groups, all layers)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out / "mobilenetv2_cifar10_5_quality_vs_robustness.png")


# ======================================================================================== Stage 5 look (SURF, COL, blue / orange heatmaps)
def annotate(ax, M, fmt, norm, thr=0.6, fs=7.4):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=fs, color="white" if norm(v) > thr else INK)


def sci(v):
    return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def s5_protection(d, out):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.6), gridspec_kw={"top": 0.78, "bottom": 0.2, "left": 0.06, "right": 0.985, "wspace": 0.28})
    title(fig, "Protection on the stored 32-bit codewords: what is detected, what is corrected, what it costs",
          "100 sampled codewords per layer, 53 layers x 7 formats (36,916 codewords per bar). SECDED fixes every single flip and flags every double flip;\nparity and DMR only detect. Detected-but-uncorrected words are consumed as read, so the error stays residual.")
    strat = ["none", "parity", "secded", "dmr"]
    lab = ["none", "parity", "SECDED", "DMR"]
    for a, sc, ttl in zip(ax[:2], ("single_data", "double_data"), ("one flipped data bit per codeword", "two flipped data bits per codeword")):
        rows = {r["strategy"]: r for r in d.prot if r["format_key"] == "ALL" and r["region"] == "payload" and r["scenario"] == sc}
        w = 0.26
        for i, (k, c) in enumerate([("detection_rate", "#2a78d6"), ("correction_rate", "#1baf7a"), ("residual_word_error_rate", "#e34948")]):
            a.bar(np.arange(4) + (i - 1) * w, [f(rows[s][k]) * 100 for s in strat], w, color=c, label=k.replace("_rate", "").replace("residual_word_error", "residual word error"))
        a.set_xticks(range(4), lab)
        a.set_ylim(0, 108)
        a.set_ylabel("% of sampled codewords")
        a.set_title(ttl, fontsize=10.5)
        a.grid(axis="y", color=GRID, lw=0.8)
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9)
    for i, s in enumerate(strat):
        v = np.mean([f(r["actual_overhead_percent"]) for r in d.pstor if r["strategy"] == s])
        ax[2].bar(i, v, color=COL["e4m11_sn"], width=0.6)
        ax[2].text(i, v + 2, f"{v:.1f} %", ha="center", fontsize=9)
    ax[2].set_xticks(range(4), lab)
    ax[2].set_ylabel("extra bytes, % of the unprotected size")
    ax[2].set_title("Actual storage overhead (mean over the 7 formats)", fontsize=10.5)
    ax[2].grid(axis="y", color=GRID, lw=0.8)
    save(fig, out / "mobilenetv2_cifar10_s5_1_protection_outcomes_and_overhead.png")


def s6_overview(d, out):
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    title(fig, "Stage 6 in one picture: from measurements to a mixed datatype + protection network",
          "Every layer of MobileNetV2 (53 Conv/Linear layers) gets one of 7 stored formats and one protection. The score tables are estimates, the assembled model is the measurement.")

    def box(x, y, w, h, head, body, ec=GRID):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc="#ffffff", ec=ec, lw=1.4))
        ax.text(x + 1.2, y + h - 1.6, head, fontsize=11.5, fontweight="bold", va="top", color=INK)
        ax.text(x + 1.2, y + h - 4.6, body, fontsize=9.3, va="top", color=INK2, linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))

    box(1, 36, 28, 14, "Q(l,d)  quantization penalty", "Clean loss layer l adds when stored in\nformat d:  sum Fisher x (w - stored w)^2\n(exact Fisher, validation images 0-2047).", ec=COL["e4m11_sn"])
    box(1, 19, 28, 14, "R(l,d,p)  residual fault risk", "Stage 4 accuracy drop per flipped bit x\nthe layer's weight share x residual rate\nof the protection p (SECDED 0).", ec=COL["fp16"])
    box(1, 2, 28, 14, "S(l,d,p)  storage", "Actual stored bytes of layer l: payload +\nscale + padding + check bits / copies\n(parity 3 %, SECDED 22 %, DMR 100 %).", ec=COL["fp8_e4m3fn"])
    for y in (43, 26, 9):
        arrow(29.8, y, 34.4, 26)
    box(35, 8, 29, 36, "Optimiser (exact ILP)", "choose ONE (datatype, protection)\nper layer, 30 options each\n\nminimise  sum Q_norm + lambda R_norm\n\nsubject to\n  total bytes <= B = 8 bit/weight\n  predicted sum R <= tau_fault\n  scale-word risk <= tau_scale\n  (clean accuracy and measured faults\n   are checked on the assembled model)\n\nsweep: 1,440 problems\n-> 201 distinct assignments", ec=INK)
    arrow(64.8, 26, 69.4, 26)
    box(70, 31, 29, 19, "Assemble the full network", "Each layer stored in its chosen format\nwith its protection (real encoded bytes;\nbiases and BatchNorm stay FP32).", ec=INK)
    arrow(84.5, 30.6, 84.5, 25.8)
    box(70, 8, 29, 17, "Measure end to end", "clean accuracy (2,952 images not used for Q),\nstored bytes, bit-flip trials: 8 flips over\nall weights, 8 over all stored bits, and one\nscale-word flip; then freeze and test once", ec=INK)
    arrow(84.5, 7.6, 84.5, 4.4)
    ax.text(84.5, 2.4, "final: A000, 91.62 % test accuracy\n(FP32 91.69 %), 5.996 bit/weight", fontsize=9.4, color=COL["fp16"], fontweight="bold", va="center", ha="center")
    save(fig, out / "mobilenetv2_cifar10_s6_1_overview.png")


def s6_profile(d, out):
    L = len(d.layers)
    y = np.arange(L)
    nw = np.array([d.nw[l] for l in d.layers]) / 1e6
    tr = {r["layer_name"]: f(r["fisher_trace"]) for r in d.fq if r["format_key"] == "int4_pt"}
    trace = np.array([tr[l] / d.nw[l] for l in d.layers])
    pred = np.array([f(d.tab[(l, "int4_pt", "none")]["Q_pred_dCE_fisher"]) for l in d.layers])
    meas = np.array([d.s3m[(l, "int4_pt")] for l in d.layers])
    fig, axs = plt.subplots(1, 4, figsize=(15, 13), sharey=True, gridspec_kw={"wspace": 0.12, "top": 0.9, "bottom": 0.04, "left": 0.085, "right": 0.985})
    title(fig, "How sensitive is each layer?  (MobileNetV2, 53 Conv/Linear layers, input side at the top)",
          "Columns b-d use INT4 as the stress test, because only INT4 changes accuracy measurably. b = block, exp = 1x1 expansion, dw = depthwise 3x3, proj = 1x1 projection.", y=0.992)
    blue = COL["e4m11_sn"]
    axs[0].barh(y, nw, color=blue, height=0.62)
    axs[0].set_title("a  Weights in the layer (millions)", fontsize=10.5)
    for yi, v in zip(y, nw):
        axs[0].text(v + 0.004, yi, f"{v:.3f}" if v < 0.01 else f"{v:.2f}", va="center", fontsize=7, color=INK2)
    axs[1].barh(y, trace, color=blue, height=0.62)
    axs[1].set_xscale("log")
    axs[1].set_title("b  Fisher sensitivity\n(loss curvature per weight)", fontsize=10.5)
    axs[2].barh(y, pred, color=blue, height=0.62)
    axs[2].set_xscale("log")
    axs[2].set_title("c  Predicted loss increase if only\nthis layer is INT4  (1/2 Q)", fontsize=10.5)
    pos = meas > 0
    axs[3].barh(y[pos], meas[pos], color=blue, height=0.62)
    axs[3].barh(y[~pos], meas[~pos], color="#b9b8b2", height=0.62)
    axs[3].axvline(0, color=INK2, lw=0.8)
    axs[3].set_title("d  Measured loss change, only this\nlayer INT4 (5,000 val. images)", fontsize=10.5)
    axs[3].text(0.97, 0.02, "grey = quantization lowered the loss\n(not 'more sensitive')", fontsize=8.4, color=INK2, va="bottom", ha="right", transform=axs[3].transAxes)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([short(l) for l in d.layers], fontsize=8)
    axs[0].invert_yaxis()
    for a in axs:
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.tick_params(axis="y", length=0)
    save(fig, out / "mobilenetv2_cifar10_s6_2_layer_sensitivity_profile.png")


def s6_Q(d, out):
    Q = np.array([[f(d.tab[(l, k, "none")]["Q_pred_dCE_fisher"]) for k in FORMATS] for l in d.layers])
    fig, ax = plt.subplots(figsize=(11.5, 15.5))
    fig.subplots_adjust(top=0.9, bottom=0.04, left=0.12, right=0.87)
    title(fig, "Q: predicted clean-loss increase per layer and format",
          "1/2 x sum of Fisher x (weight - stored weight)^2.  Darker = this layer loses more accuracy in this format.\nThe format sets the size (INT4 >> FP8 E5M2 > FP8 E4M3 >> INT8 >> BF16 >> FP16 > E4M11); the layer sets the pattern inside each column.", y=0.992)
    vmin = 1e-12
    norm = LogNorm(vmin=vmin, vmax=max(1e-1, np.nanmax(Q)))
    Qc = np.maximum(Q, vmin)
    im = ax.imshow(Qc, cmap=SEQ_BLUE, norm=norm, aspect="auto")
    annotate(ax, Qc, sci, lambda v: norm(v), 0.55, 7)
    ax.set_xticks(range(len(FORMATS)))
    ax.set_xticklabels([f"{LABEL[k]}\n{BITS[k]} bit" for k in FORMATS], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(d.layers)))
    ax.set_yticklabels([short(l) for l in d.layers], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.89, 0.3, 0.022, 0.4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("predicted loss increase (log scale)", fontsize=9)
    save(fig, out / "mobilenetv2_cifar10_s6_3_Q_quantization_penalty_heatmap.png")


def s6_R(d, out):
    R = np.array([[8 * f(d.tab[(l, k, "none")]["per_flip_drop_pp"]) for k in FORMATS] for l in d.layers])
    scl = {(r["layer_name"], r["format_key"]): f(r["mean_drop_pp"]) for r in d.scale}
    RS = np.array([[scl[(l, k)] for k in ("int8_pt", "int4_pt")] for l in d.layers])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 15.5), gridspec_kw={"width_ratios": [7, 2.3], "wspace": 0.3, "top": 0.86, "bottom": 0.07, "left": 0.07, "right": 0.95})
    title(fig, "R: accuracy lost when bits flip in ONE layer  (Stage 4 screening, percentage points)",
          "Left: 8 flips among the stored weights of that one layer = 8 x the per-flip drop of the Stage 4 layer-wise trials (10 trials x 1,000 images, k = 1 and 8 combined, group-weighted).\n"
          "Right: one random bit of the layer's float32 scale (INT only; exhaustive over the 32 bits). Colour is clipped at the top of the bar; values below 0 are noise.", y=0.992)
    n1 = Normalize(vmin=0, vmax=20, clip=True)
    im1 = a1.imshow(np.clip(R, 0, None), cmap=SEQ_BLUE, norm=n1, aspect="auto")
    annotate(a1, R, lambda v: f"{v:.2f}", lambda v: n1(max(v, 0)), 0.55, 7)
    a1.set_xticks(range(len(FORMATS)))
    a1.set_xticklabels([LABEL[k] for k in FORMATS], fontsize=9)
    a1.set_title("Weight bits, any bit, 8 flips in the layer", fontsize=10.5, pad=24)
    n2 = Normalize(vmin=0, vmax=60, clip=True)
    im2 = a2.imshow(RS, cmap=SEQ_ORANGE, norm=n2, aspect="auto")
    annotate(a2, RS, lambda v: f"{v:.1f}", lambda v: n2(v), 0.55, 7)
    a2.set_xticks(range(2))
    a2.set_xticklabels(["INT8\nscale word", "INT4\nscale word"], fontsize=9)
    a2.set_title("Scale word, mean of 32 bits", fontsize=10.5, pad=24)
    for a in (a1, a2):
        a.xaxis.tick_top()
        a.set_yticks(range(len(d.layers)))
        a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
    a1.set_yticklabels([short(l) for l in d.layers], fontsize=8)
    a2.set_yticklabels([])
    for im, a, lab in ((im1, a1, "mean accuracy drop (pp), colour clipped at 20"), (im2, a2, "mean accuracy drop (pp), clipped at 60")):
        pos = a.get_position()
        cax = fig.add_axes([pos.x0, 0.035, pos.width, 0.008])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=8.5)
    save(fig, out / "mobilenetv2_cifar10_s6_4_R_fault_risk_heatmaps.png")


def s6_Qval(d, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"wspace": 0.28, "top": 0.82, "bottom": 0.12, "left": 0.07, "right": 0.975})
    title(fig, "Does the sensitivity score Q predict what really happens?",
          "Left: quantize ONE layer (INT4) and measure the validation loss change (Stage 3, 5,000 images). Right: quantize ALL layers in one format (Stage 2, 5,000 images). The Fisher uses images 0-2047, which these sets include.")
    px = np.array([f(d.tab[(l, "int4_pt", "none")]["Q_pred_dCE_fisher"]) for l in d.layers])
    py = np.array([d.s3m[(l, "int4_pt")] for l in d.layers])
    a.axhline(0, color=GRID, lw=1)
    lim = max(px.max(), py.max()) * 1.1
    a.plot([0, lim], [0, lim], color=MUTED, lw=1.2, ls="--")
    a.text(lim * 0.45, lim * 0.52, "perfect prediction (dashed)", color=MUTED, fontsize=8.5)
    a.scatter(px, py, s=40, color=COL["int4_pt"], edgecolor=SURF, linewidth=1.0, zorder=3)
    for n, x, yv in zip(d.layers, px, py):
        if x > 0.1 * px.max() or abs(yv) > 0.1 * abs(py).max():
            a.annotate(short(n), (x, yv), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK2)
    a.set_xlabel("predicted loss increase 1/2 Q  (INT4, one layer at a time)")
    a.set_ylabel("measured loss change")
    a.set_title(f"INT4, 53 layers: Spearman {spearmanr(px, py)[0]:+.2f}", fontsize=10.5)
    a.grid(color=GRID, lw=0.8)
    pred = {k: 0.5 * sum(f(r["Q_fisher_val_exact"]) for r in d.fq if r["format_key"] == k) for k in FORMATS}
    meas = {k: f(d.clean[k]["val_delta_ce"]) for k in FORMATS}
    order = list(reversed(FORMATS))
    b.set_xscale("symlog", linthresh=1e-5)
    b.axvline(0, color=GRID, lw=1)
    for i, k in enumerate(order):
        b.plot([pred[k], meas[k]], [i, i], color=GRID, lw=3, zorder=1)
        b.scatter([pred[k]], [i], s=70, facecolor=SURF, edgecolor=COL[k], linewidth=2.2, zorder=3)
        b.scatter([meas[k]], [i], s=70, color=COL[k], zorder=3)
    b.set_yticks(range(len(order)))
    b.set_yticklabels([LABEL[k] for k in order])
    b.set_xlabel("loss change when ALL layers use the format (symmetric log scale)")
    b.set_xlim(-3e-3, 1.0)
    b.grid(axis="x", color=GRID, lw=0.8)
    b.set_title("Whole network: predicted (hollow) vs measured (filled)", fontsize=10.5)
    b.legend(handles=[Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, mew=2, ms=8, label="predicted  sum 1/2 Q"), Line2D([], [], marker="o", ls="", color=INK2, ms=8, label="measured")],
             loc="lower right", frameon=False, fontsize=9)
    save(fig, out / "mobilenetv2_cifar10_s6_5_Q_validation_predicted_vs_measured.png")


def s6_assign(d, out):
    cols = ["U_INT4_TMR", "A006", "A007", "A058", "A000", "A021", "U_INT8_TMR", "U_INT8_SECDED"]
    names = {"U_INT4_TMR": "uniform INT4\n+ scale TMR", "U_INT8_TMR": "uniform INT8\n+ scale TMR", "U_INT8_SECDED": "uniform INT8\n+ SECDED", "A000": "A000 FINAL"}
    L = len(d.layers)
    fig = plt.figure(figsize=(15, 17))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 9], wspace=0.04, top=0.80, bottom=0.04, left=0.07, right=0.985)
    a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    title(fig, "Which datatype and protection does each layer get?  Selected assignments, cheapest on the left",
          "The final assignment A000 is outlined. Cell text = datatype, then +S = SECDED, +T = scale TMR (INT only). Late, large 1x1 layers are the least sensitive and go to INT4;\n"
          "the small early layers stay in E4M11. Column header: average stored bits per weight / clean accuracy change vs FP32 (pp) / mean drop under 8 random weight-bit flips over the whole model (pp).", y=0.992)
    pred = np.array([f(d.tab[(l, "int4_pt", "none")]["Q_pred_dCE_fisher"]) for l in d.layers])
    a0.barh(range(L), pred, color=COL["e4m11_sn"], height=0.62)
    a0.set_xscale("log")
    a0.set_ylim(L - 0.5, -0.5)
    a0.set_yticks(range(L))
    a0.set_yticklabels([short(l) for l in d.layers], fontsize=8)
    a0.set_title("sensitivity\n(1/2 Q, INT4)", fontsize=9.5, pad=70)
    a0.grid(axis="x", color=GRID, lw=0.8)
    a0.tick_params(axis="y", length=0)
    a1.set_xlim(-0.5, len(cols) - 0.5)
    a1.set_ylim(L - 0.5, -0.5)
    for j, aid in enumerate(cols):
        a = d.A[aid]
        for i, o in enumerate(a["options"]):
            fm, pr = o.split("+")[0], (o.split("+")[1] if "+" in o else "")
            a1.add_patch(plt.Rectangle((j - 0.47, i - 0.46), 0.94, 0.92, fc=COL[fm], ec=SURF, lw=1))
            a1.text(j, i, SHORTF[fm] + {"secded": "+S", "scale_tmr": "+T", "parity": "+P", "dmr": "+D"}.get(pr, ""), ha="center", va="center", fontsize=7.5,
                    color="white" if fm not in ("fp8_e4m3fn", "bf16", "fp8_e5m2") else INK, fontweight="bold")
        w = a["fault_end_to_end"]["weights_k8"]["drop_pp_mean"]
        a1.text(j, -0.9, f"{names.get(aid, aid)}\n{a['storage']['avg_bits_per_weight']:.1f} bit\nclean {-a['clean_validation']['drop_pp_vs_fp32']:+.2f} pp\nfaults {w:.2f} pp", ha="center", va="bottom", fontsize=8.8, color=INK,
                 linespacing=1.4, fontweight="bold" if aid == "A000" else "normal")
        if aid == "A000":
            a1.add_patch(plt.Rectangle((j - 0.5, -0.5), 1.0, L, fill=False, ec=INK, lw=2.2, zorder=5))
    a1.set_yticks([])
    a1.set_xticks([])
    for s in a1.spines.values():
        s.set_visible(False)
    a1.legend(handles=[Patch(fc=COL[k], label=f"{LABEL[k]} ({BITS[k]} bit)") for k in FORMATS], ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=8.5)
    save(fig, out / "mobilenetv2_cifar10_s6_6_selected_assignments_per_layer.png")


def s6_tradeoff(d, out):
    items = d.m["assignments"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw={"wspace": 0.2, "top": 0.8, "bottom": 0.12, "left": 0.06, "right": 0.985})
    title(fig, "Storage, clean accuracy and fault robustness of every assembled assignment",
          "Left: all 212 assembled models (grey = optimiser, squares = uniform references). Right: the 33 with fault trials (100 trials, 8 random weight-bit flips). Dashed = limits delta_clean and tau_fault (0.5 pp).")
    fin = d.e2e["A000"]
    for it in items:
        aid = it["assignment_id"]
        if not aid.startswith("U_"):
            ax[0].scatter(it["storage"]["avg_bits_per_weight"], it["clean_validation"]["drop_pp_vs_fp32"], s=26, color="#b9b8b2", zorder=2, edgecolor=SURF, linewidth=0.6)
    for it in items:
        aid = it["assignment_id"]
        if aid.startswith("U_"):
            x, y = it["storage"]["avg_bits_per_weight"], it["clean_validation"]["drop_pp_vs_fp32"]
            ax[0].scatter(x, y, s=48, marker="s", color=INK2, zorder=3)
            ax[0].annotate(aid[2:], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK2)
    ax[0].scatter(f(fin["avg_bits"]), f(fin["clean_drop_pp"]), s=170, marker="*", color=COL["int4_pt"], zorder=4, edgecolor=INK, linewidth=0.8)
    ax[0].annotate("A000 (final)", (f(fin["avg_bits"]), f(fin["clean_drop_pp"])), xytext=(8, 10), textcoords="offset points", fontsize=9, fontweight="bold")
    ax[0].axhline(0.5, color=INK, ls="--", lw=1)
    ax[0].set_yscale("symlog", linthresh=0.1)
    ax[0].set_ylim(-0.25, 12)
    ax[0].set_xlabel("stored bits per weight")
    ax[0].set_ylabel("clean accuracy drop vs FP32, pp (2,952 images)")
    ax[0].set_title("Accuracy cost", fontsize=10.5)
    ax[0].grid(color=GRID, lw=0.8)
    for it in items:
        fa = it["fault_end_to_end"].get("weights_k8")
        if not fa:
            continue
        ref = it["assignment_id"].startswith("U_")
        ax[1].scatter(it["storage"]["avg_bits_per_weight"], fa["drop_pp_mean"], s=48 if ref else 30, marker="s" if ref else "o", color=INK2 if ref else COL["e4m11_sn"], zorder=3,
                      edgecolor=SURF, linewidth=0.6)
        if ref:
            ax[1].annotate(it["assignment_id"][2:], (it["storage"]["avg_bits_per_weight"], fa["drop_pp_mean"]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK2)
    ax[1].scatter(f(fin["avg_bits"]), f(fin["weights_k8_drop_pp_mean"]), s=170, marker="*", color=COL["int4_pt"], zorder=4, edgecolor=INK, linewidth=0.8)
    ax[1].axhline(0.5, color=INK, ls="--", lw=1)
    ax[1].set_yscale("symlog", linthresh=0.1)
    ax[1].set_ylim(-0.25, 40)
    ax[1].set_xlabel("stored bits per weight")
    ax[1].set_ylabel("mean drop under 8 random weight-bit flips, pp")
    ax[1].set_title("Fault cost", fontsize=10.5)
    ax[1].grid(color=GRID, lw=0.8)
    save(fig, out / "mobilenetv2_cifar10_s6_7_storage_accuracy_fault_tradeoff.png")


def s6_predmeas(d, out):
    it = d.m["assignments"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"wspace": 0.25, "top": 0.8, "bottom": 0.12, "left": 0.07, "right": 0.975})
    title(fig, "Layer-additive estimates against the assembled models",
          "The optimiser predicts with sums over layers; the final decision uses the measured end-to-end values.")
    x = np.array([i["predicted_layer_additive"]["sum_Q"] * 0.5 for i in it])
    y = np.array([i["clean_validation"]["delta_ce_vs_fp32"] for i in it])
    ax[0].scatter(x, y, s=24, color=COL["e4m11_sn"], edgecolor=SURF, linewidth=0.6)
    ax[0].set_xscale("symlog", linthresh=1e-4)
    ax[0].set_yscale("symlog", linthresh=1e-3)
    ax[0].set_title(f"Clean loss change (212 models), Spearman {spearmanr(x, y)[0]:+.2f}", fontsize=10.5)
    ax[0].set_xlabel("predicted: 1/2 x sum Q")
    ax[0].set_ylabel("measured loss change vs FP32")
    ax[0].grid(color=GRID, lw=0.8)
    fi = [i for i in it if i["fault_end_to_end"].get("weights_k8")]
    xp = np.array([i["predicted_layer_additive"]["weight_fault_drop_pp"] for i in fi])
    yp = np.array([i["fault_end_to_end"]["weights_k8"]["drop_pp_mean"] for i in fi])
    ax[1].scatter(xp, yp, s=34, color=COL["fp16"], edgecolor=SURF, linewidth=0.6)
    lim = [min(xp.min(), yp.min(), 0) - 0.05, max(xp.max(), yp.max()) * 1.05]
    ax[1].plot(lim, lim, color=MUTED, lw=1.2, ls="--")
    ax[1].set_title(f"Weight-fault drop (33 models), Spearman {spearmanr(xp, yp)[0]:+.2f}", fontsize=10.5)
    ax[1].set_xlabel("predicted: sum of layer R, pp")
    ax[1].set_ylabel("measured mean drop, 8 random weight bits, pp")
    ax[1].grid(color=GRID, lw=0.8)
    save(fig, out / "mobilenetv2_cifar10_s6_8_predicted_vs_measured_assignments.png")


def s6_trials(d, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"wspace": 0.28, "top": 0.74, "bottom": 0.13, "left": 0.06, "right": 0.985})
    title(fig, "Fault trials on the frozen assignment A000 (validation, 300 trials per distribution)",
          "All trials kept, 0 failures, byte-exact restoration after every trial. The model is unchanged by almost every trial: the protection removes the flips before they reach the weights.")
    for a, dist, ttl in zip(ax, ("weights_k8", "stored_k8", "scales_k1"), ("8 random weight bits", "8 random stored bits (incl. check bits)", "1 random scale-word bit")):
        v = np.array([t["clean_accuracy"] - t["fault_accuracy"] for t in d.trials if t["distribution"] == dist])
        a.hist(v, bins=np.arange(-0.65, 0.66, 0.1), color=COL["e4m11_sn"], edgecolor=SURF)
        a.set_yscale("log")
        a.set_title(f"{ttl}\nmean {v.mean():+.3f} pp, worst {v.max():.2f} pp", fontsize=10.5)
        a.set_xlabel("accuracy drop, pp (1,000 images)")
        a.set_ylabel("trials (log)")
        a.grid(axis="y", color=GRID, lw=0.8)
    save(fig, out / "mobilenetv2_cifar10_s6_9_fault_trial_distributions.png")


def s6_sweep(d, out):
    rows = [r for r in d.sweep if r["mode"] == "joint" and r["normalization"] == "minmax" and f(r["B_avg_bits_per_weight"]) == 8 and r["tau_fault_pred_drop_pp"] == "" and r["status"] == "optimal"]
    rows.sort(key=lambda r: f(r["lambda"]))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"wspace": 0.25, "top": 0.78, "bottom": 0.2, "left": 0.06, "right": 0.985})
    title(fig, "Effect of lambda on the Fisher-Q optimum (joint mode, min-max, B = 8 bit/weight, no tau_fault)",
          "Datatype mix of the optimal assignment for each lambda (left) and its average stored bits (right). The full sweep (1,440 problems) is in normalization_lambda_sweep.csv.")
    bottom = np.zeros(len(rows))
    for fm in FORMATS:
        c = np.array([sum(1 for t in r["assignment"].split(";") if TOK.get(t.split("+")[0]) == fm) for r in rows])
        ax[0].bar(range(len(rows)), c, bottom=bottom, color=COL[fm], label=LABEL[fm], edgecolor=SURF)
        bottom += c
    ax[0].set_xticks(range(len(rows)), [r["lambda"] for r in rows])
    ax[0].set_xlabel("lambda")
    ax[0].set_ylabel("layers")
    ax[0].legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax[1].plot([f(r["lambda"]) for r in rows], [f(r["avg_bits_per_weight"]) for r in rows], "o-", color=COL["fp16"])
    ax[1].set_xscale("symlog", linthresh=0.1)
    ax[1].set_xlabel("lambda")
    ax[1].set_ylabel("stored bits per weight")
    ax[1].grid(color=GRID, lw=0.8)
    save(fig, out / "mobilenetv2_cifar10_s6_10_lambda_sweep_composition.png")


def s6_vs_uniform(d, out):
    ids = ["U_E4M11", "U_FP16", "U_BF16", "U_FP8N", "U_FP8E5", "U_INT8", "U_INT4", "A000"]
    keys = ["e4m11_sn", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "int8_pt", "int4_pt"]
    lab = ["E4M11", "FP16", "BF16", "FP8\nE4M3FN", "FP8\nE5M2", "INT8", "INT4", "A000\nassigned"]
    col = [COL[k] for k in keys] + [INK]
    fig, ax = plt.subplots(1, 4, figsize=(17, 5.6), gridspec_kw={"wspace": 0.28, "top": 0.74, "bottom": 0.13, "left": 0.05, "right": 0.99})
    title(fig, "The assigned model against the uniform, unprotected models (every layer in one format, no protection)",
          "Same assembled-model protocol (validation images 2,048-4,999; 100 fault trials). Scale-word faults exist only for INT. A000 protects all 53 layers and is the only one that passes every limit with 6 bit/weight.")
    spec = [("avg_bits", "stored bits per weight", None), ("clean_val_accuracy", "clean validation accuracy, %", (82, 93)), ("weights_k8_drop_pp_mean", "drop, 8 random weight bits, pp", "sym"),
            ("scales_k1_drop_pp_mean", "drop, 1 scale-word bit, pp (INT only)", "sym")]
    for a, (k, ttl, lim) in zip(ax, spec):
        y = [f(d.e2e[i][k]) for i in ids]
        a.bar(range(8), np.nan_to_num(y), color=col, width=0.7)
        for i, v in enumerate(y):
            a.text(i, (0 if np.isnan(v) else v) + (0.2 if lim != "sym" else 0), "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        a.set_xticks(range(8), lab, fontsize=7.5)
        a.set_title(ttl, fontsize=10)
        a.grid(axis="y", color=GRID, lw=0.8)
        if lim == "sym":
            a.set_yscale("symlog", linthresh=0.1)
            a.set_ylim(0, max(np.nan_to_num(y)) * 3)
            a.axhline({"weights_k8_drop_pp_mean": 0.5, "scales_k1_drop_pp_mean": 1.0}[k], color=INK, ls="--", lw=1)
        elif lim:
            a.set_ylim(*lim)
    fp32 = f(d.m["fp32_reference"]["val_accuracy"])
    ax[1].axhline(fp32, color=INK, ls="--", lw=1)
    save(fig, out / "mobilenetv2_cifar10_s6_11_assigned_vs_uniform.png")


def s6_valtest(d, out):
    fig, ax = plt.subplots(figsize=(10, 5.8), gridspec_kw={"top": 0.78, "bottom": 0.14, "left": 0.09, "right": 0.97})
    title(fig, "Official test accuracy of the frozen assignments (10,000 images, read once per variant)",
          "Both assignments were frozen from validation data before the test set was read. The variant reported as main result was not chosen by test accuracy.")
    t = d.test
    lab = ["FP32", "Fisher Q\n(A000, main)", "measured Q\n(comparison)"]
    test = [t["fp32_test_accuracy"], t["assembled_test_accuracy"], d.testA["assembled_test_accuracy"]]
    ax.bar(range(3), test, color=[MUTED, COL["e4m11_sn"], "#b9b8b2"], width=0.55)
    for i, v in enumerate(test):
        ax.text(i, v + 0.05, f"{v:.2f} %", ha="center", fontsize=10)
    ax.set_ylim(90, 92.5)
    ax.set_xticks(range(3), lab)
    ax.set_ylabel("test accuracy, %")
    ax.grid(axis="y", color=GRID, lw=0.8)
    save(fig, out / "mobilenetv2_cifar10_s6_12_validation_vs_test_and_variants.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", default="reports/mobilenetv2_cifar10")
    ap.add_argument("--out", default="reports/mobilenetv2_cifar10/figures")
    a = ap.parse_args()
    out = Path(a.out)
    d = D(a.reports)
    print("figures ->", out)
    with plt.rc_context(RC4):
        print(" stage4 look")
        for fn in (s4_overview, s4_global, s4_scales, s4_heatmaps, s4_quality):
            fn(d, out / "stage4")
    with plt.rc_context(RC5):
        print(" stage5/6 look")
        s5_protection(d, out / "stage5_protection")
        for fn in (s6_overview, s6_profile, s6_Q, s6_R, s6_Qval, s6_assign, s6_tradeoff, s6_predmeas, s6_trials, s6_sweep, s6_vs_uniform, s6_valtest):
            fn(d, out / "stage6")


if __name__ == "__main__":
    main()
