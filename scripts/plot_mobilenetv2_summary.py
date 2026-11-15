#!/usr/bin/env python
"""MobileNetV2 / CIFAR-10 figures for Stage 4 (encoded faults), the protection study (Section 5) and Stage 6 (Fisher-Q optimisation, final assignment).

Runs on the committed CSV / JSON files only (no model, no data, no GPU):

  python scripts/plot_mobilenetv2_summary.py --reports reports/mobilenetv2_cifar10 --out reports/mobilenetv2_cifar10/figures

The figure TYPES follow the ResNet-18 plot scripts (plot_stage4_summary.py: format overview, global screening, layer heatmap, quality vs robustness; plot_stage5_summary.py /
stage6_final_package.py: Q / R heatmaps, Q validation, selected assignment, trade-off, predicted vs measured, fault-trial distributions, lambda sweep, validation vs test).
NOT reproduced: the ResNet figures of weight / activation / logit distributions and activation growth (they need per-trial tensors that the MobileNetV2 campaigns did not save),
and the figure of the datatype-only versus protected pairs (round-1 data of ResNet-18). Colours: the reference categorical palette, first 7 slots (validated with
validate_palette.js: all hard checks pass, 3 light slots are below 3:1 contrast on the surface -> every series is also labelled directly or in a table).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from fr import stage3 as S3

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
FMT = ["fp16", "bf16", "e4m11_sn", "fp8_e4m3fn", "fp8_e5m2", "int8_pt", "int4_pt"]
NAME = {"fp16": "FP16", "bf16": "BF16", "e4m11_sn": "E4M11-SN", "fp8_e4m3fn": "FP8 E4M3FN", "fp8_e5m2": "FP8 E5M2", "int8_pt": "INT8", "int4_pt": "INT4"}
COL = dict(zip(FMT, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
BITS = {"fp16": 16, "bf16": 16, "e4m11_sn": 16, "fp8_e4m3fn": 8, "fp8_e5m2": 8, "int8_pt": 8, "int4_pt": 4}
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
ORG = LinearSegmentedColormap.from_list("org", ["#fef3ec", "#f6c3a8", "#eb6834", "#b0451a", "#6e2608"])
TOK = {"FP16": "fp16", "BF16": "bf16", "E4M11": "e4m11_sn", "FP8N": "fp8_e4m3fn", "FP8E5": "fp8_e5m2", "INT8": "int8_pt", "INT4": "int4_pt"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 100})


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def f(x, d=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def save(fig, out, name, main, sub):
    fig.suptitle(main, x=0.01, ha="left", fontsize=12, fontweight="bold", y=0.995)
    import textwrap
    wrapped = textwrap.fill(sub, width=int(fig.get_figwidth() * 13))
    fig.text(0.01, 0.955, wrapped, ha="left", fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.92 if chr(10) in wrapped else 0.93))
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print("  ", name)


def short(n):
    return n.replace("features.", "f").replace(".conv.", ".").replace("classifier.1", "cls")


class D:
    def __init__(self, rep):
        r = Path(rep)
        self.r = r
        self.clean = {x["format_key"]: x for x in rd(r / "jfamp/quantization/clean_accuracy.csv")}
        self.qm = rd(r / "jfamp/quantization/quantization_metrics.csv")
        self.stor = {x["format_key"]: x for x in rd(r / "jfamp/quantization/storage_summary.csv")}
        self.gs = rd(r / "stage_04_encoded_fault_injection/global_summary.csv")
        self.cand = rd(r / "stage_06_fisher/jfamp_candidate_table.csv")
        self.s3 = rd(r / "stage_03_layerwise_sensitivity/layerwise_sensitivity_results.csv")
        self.prot = rd(r / "stage_05_protection/protection_summary.csv")
        self.pstor = rd(r / "stage_05_protection/protection_storage_summary.csv")
        self.scale = rd(r / "stage_06_optimization/scale_faults/scale_summary.csv")
        self.e2e = {x["assignment_id"]: x for x in rd(r / "stage_06_fisher/end_to_end_assignments.csv")}
        self.m = json.load(open(r / "stage_06_fisher/end_to_end_assignment_metrics.json"))
        self.asg = json.load(open(r / "stage_06_fisher/assignment.json"))
        self.test = json.load(open(r / "stage_06_fisher/final_test_evaluation.json"))
        self.sweep = rd(r / "stage_06_fisher/normalization_lambda_sweep.csv")
        self.trials = [json.loads(l) for l in open(r / "stage_06_fisher/final_validation_fault_trials.jsonl", encoding="utf-8")]
        self.layers = [x["layer_name"] for x in self.asg["selected"]]
        self.nw = {x["layer_name"]: int(x["num_weights"]) for x in self.asg["selected"]}


# ------------------------------------------------------------------------------------------------ Stage 4 (format quality and encoded-fault screening)
def fig_format_overview(d, out):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    fp32 = f(d.clean["fp32"]["val_accuracy_percent"])
    v = [f(d.clean[k]["val_accuracy_percent"]) for k in FMT]
    ax[0].bar(range(7), v, color=[COL[k] for k in FMT], width=0.7)
    ax[0].axhline(fp32, color=INK, lw=1, ls="--")
    ax[0].text(6.4, fp32 + 0.15, f"FP32 {fp32:.2f} %", ha="right", fontsize=8)
    ax[0].set_ylim(80, 93.5)
    ax[0].set_title("Clean validation accuracy (5,000 images)")
    ax[0].set_ylabel("accuracy, %")
    for i, y in enumerate(v):
        ax[0].text(i, y + 0.15, f"{y:.2f}", ha="center", fontsize=7.5)
    sq = {k: [f(x["sqnr_db"]) for x in d.qm if x["format_key"] == k and x["sqnr_db"] not in ("", "inf")] for k in FMT}
    med = [float(np.nanmedian([s for s in sq[k] if np.isfinite(s)])) for k in FMT]
    ax[1].bar(range(7), med, color=[COL[k] for k in FMT], width=0.7)
    ax[1].set_title("Weight SQNR, median over 53 layers")
    ax[1].set_ylabel("dB")
    for i, y in enumerate(med):
        ax[1].text(i, y + 0.8, f"{y:.0f}", ha="center", fontsize=7.5)
    bpw = [f(d.stor[k]["actual_bits_per_weight"]) for k in FMT]
    ax[2].bar(range(7), bpw, color=[COL[k] for k in FMT], width=0.7)
    ax[2].set_title("Stored bits per weight (payload + scales)")
    ax[2].set_ylabel("bits / weight (FP32 = 32)")
    for i, y in enumerate(bpw):
        ax[2].text(i, y + 0.3, f"{y:.2f}", ha="center", fontsize=7.5)
    for a in ax:
        a.set_xticks(range(7), [NAME[k] for k in FMT], rotation=35, ha="right")
    save(fig, out, "01_format_overview.png", "Quantization quality of the seven formats (MobileNetV2, CIFAR-10)", "Stage 2 frozen encodings; INT policy per-tensor; accuracy on the 5,000 validation images")


def fig_global_screening(d, out):
    grp = [("sign", "#52514e", ""), ("exponent", "#c0392b", "//"), ("fraction", "#2a78d6", ".."), ("value", "#2a78d6", "..")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for a, k in zip(ax, (1, 8)):
        w = 0.26
        for gi, (g, c, h) in enumerate([("sign", "#8a8984", ""), ("exponent", "#e34948", "//"), ("fraction/value", "#2a78d6", "..")]):
            ys = []
            for fm in FMT:
                gg = "value" if g == "fraction/value" and BITS[fm] <= 8 and fm.startswith("int") else ("fraction" if g == "fraction/value" else g)
                row = [r for r in d.gs if r["format"] == fm and r["fault_group"] == gg and int(r["k"]) == k]
                ys.append(f(row[0]["drop_pp_mean"]) if row else np.nan)
            a.bar(np.arange(7) + (gi - 1) * w, ys, w, color=c, hatch=h, edgecolor=SURF, label=g)
        a.set_xticks(range(7), [NAME[x] for x in FMT], rotation=35, ha="right")
        a.set_title(f"k = {k} flipped bit{'s' if k > 1 else ''} (30 trials per bar)")
    ax[0].set_ylabel("mean accuracy drop, pp (1,000-image subset)")
    ax[0].legend(title="fault group", loc="upper left")
    save(fig, out, "02_global_screening_weights.png", "Global screening: random bit flips in the stored encoded weights",
         "Stage 4: flips on the stored bits over all 53 layers; exponent bits dominate the float formats; INT formats have no exponent group")


def fig_scale_faults(d, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 2.2]})
    for i, k in enumerate(("int8_pt", "int4_pt")):
        v = [f(r["mean_drop_pp"]) for r in d.scale if r["format_key"] == k]
        ax[0].scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, s=14, color=COL[k], alpha=0.8)
        ax[0].hlines(np.mean(v), i - 0.3, i + 0.3, color=INK, lw=2)
        ax[0].text(i + 0.32, np.mean(v), f"mean {np.mean(v):.1f}", ha="left", va="center", fontsize=8)
        ax[0].set_xlim(-0.6, 1.9)
    ax[0].axhline(1.0, color=INK, ls="--", lw=1)
    ax[0].text(1.45, 1.6, "tau_scale = 1 pp", ha="right", fontsize=8)
    ax[0].set_xticks([0, 1], ["INT8", "INT4"])
    ax[0].set_ylabel("mean drop over the 32 scale-word bits, pp")
    ax[0].set_title("One point = one INT layer")
    for k, c in (("int8_pt", COL["int8_pt"]), ("int4_pt", COL["int4_pt"])):
        rs = sorted([r for r in d.scale if r["format_key"] == k], key=lambda r: int(r["layer_index"]))
        ax[1].plot([int(r["layer_index"]) for r in rs], [f(r["mean_drop_pp_exponent"]) for r in rs], color=c, lw=1.4, label=f"{NAME[k]} exponent bits (8)")
        ax[1].plot([int(r["layer_index"]) for r in rs], [f(r["mean_drop_pp_fraction"]) for r in rs], color=c, lw=1.0, ls=":", label=f"{NAME[k]} fraction bits (23)")
    ax[1].set_xlabel("layer index")
    ax[1].set_ylabel("mean drop, pp")
    ax[1].set_title("Scale-word faults by bit group (exhaustive: all 32 bits x every INT layer)")
    ax[1].legend(fontsize=7.5, ncol=2)
    save(fig, out, "03_scale_word_faults.png", "A single flipped bit of the float32 scale of an INT layer collapses the model",
         "Stage 6 scale campaign: all 106 INT layer/format cases exceed 1 pp, so INT layers must be protected (SECDED or scale TMR)")


def heat(ax, M, rows, cols, cmap, norm, label, fmt="{:.1e}", thr=None):
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)), rows, fontsize=6)
    ax.grid(False)
    return im


def fig_layer_heat(d, out):
    pf = np.full((53, 7), np.nan)
    for r in d.cand:
        if r["protection"] == "none":
            pf[d.layers.index(r["layer_name"]), FMT.index(r["datatype"])] = f(r["per_flip_drop_pp"])
    fig, ax = plt.subplots(figsize=(7.5, 11))
    im = heat(ax, np.where(pf > 0, pf, np.nan), [short(x) for x in d.layers], [NAME[k] for k in FMT], SEQ, LogNorm(vmin=max(1e-3, np.nanmin(pf[pf > 0])), vmax=np.nanmax(pf)), "")
    fig.colorbar(im, ax=ax, shrink=0.5, label="mean accuracy drop per flipped bit, pp (log)")
    ax.set_title("")
    save(fig, out, "04_layer_fault_sensitivity_heatmap.png", "Which layers lose the most accuracy per flipped stored bit",
         "Stage 4 layer-wise screening, group-weighted per-flip drop (k=1 and k=8 combined, as used by the optimiser); blank = no positive drop measured")


def fig_quality_vs_robust(d, out):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for fm in FMT:
        sq = [f(x["sqnr_db"]) for x in d.qm if x["format_key"] == fm]
        sq = [s for s in sq if np.isfinite(s)]
        tot, wsum = 0.0, 0.0
        for g in {"fp16": ["sign", "exponent", "fraction"], "int8_pt": ["sign", "value"], "int4_pt": ["sign", "value"]}.get(fm, ["sign", "exponent", "fraction"]):
            row = [r for r in d.gs if r["format"] == fm and r["fault_group"] == g and int(r["k"]) == 8]
            if row:
                nb = len(S3.group_bits(fm, g))
                tot += nb * f(row[0]["drop_pp_mean"])
                wsum += nb
        y = tot / wsum
        ax.scatter(np.median(sq), y, s=70, color=COL[fm], zorder=3, edgecolor=INK, linewidth=0.6)
        ax.annotate(NAME[fm], (np.median(sq), y), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("median per-layer SQNR, dB (quality)")
    ax.set_ylabel("mean drop with 8 random flips, pp (bit-weighted over groups)")
    ax.set_yscale("symlog", linthresh=0.5)
    save(fig, out, "05_quality_vs_robustness.png", "Quantization quality does not predict fault robustness", "high SQNR formats with exponent bits (BF16, FP16, E5M2) are the most fragile; INT has no exponent but a fragile scale")


# ------------------------------------------------------------------------------------------------ protection (protocol Section 5)
def fig_protection(d, out):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    strat = ["none", "parity", "secded", "dmr"]
    for a, sc, ttl in zip(ax[:2], ("single_data", "double_data"), ("one flipped data bit per codeword", "two flipped data bits per codeword")):
        rows = {r["strategy"]: r for r in d.prot if r["format_key"] == "ALL" and r["region"] == "payload" and r["scenario"] == sc}
        w = 0.26
        for i, (k, c, h) in enumerate([("detection_rate", "#2a78d6", ""), ("correction_rate", "#008300", "//"), ("residual_word_error_rate", "#e34948", "..")]):
            a.bar(np.arange(4) + (i - 1) * w, [f(rows[s][k]) * 100 for s in strat], w, color=c, hatch=h, edgecolor=SURF, label=k.replace("_rate", "").replace("_word_error", " (word error)"))
        a.set_xticks(range(4), ["none", "parity", "SECDED", "DMR"])
        a.set_ylim(0, 108)
        a.set_ylabel("% of sampled codewords")
        a.set_title(ttl)
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    ov = {s: [f(r["actual_overhead_percent"]) for r in d.pstor if r["strategy"] == s] for s in strat}
    for i, s in enumerate(strat):
        v = ov[s]
        ax[2].bar(i, np.mean(v), color="#8a8984")
        ax[2].text(i, np.mean(v) + 2, f"{np.mean(v):.1f} %", ha="center", fontsize=8)
    ax[2].set_xticks(range(4), ["none", "parity", "SECDED", "DMR"])
    ax[2].set_title("Actual storage overhead (mean over formats)")
    ax[2].set_ylabel("% of unprotected bytes")
    save(fig, out, "06_protection_outcomes_and_overhead.png", "Parity, SECDED and DMR on the stored 32-bit codewords",
         "SECDED corrects every single-bit error and detects every double-bit error (detected-uncorrected words are consumed as read, so the residual stays); parity and DMR only detect")


# ------------------------------------------------------------------------------------------------ Stage 6
def fig_qr_heat(d, out):
    Q = np.full((53, 7), np.nan)
    R = np.full((53, 7), np.nan)
    for r in d.cand:
        if r["protection"] == "none":
            i, j = d.layers.index(r["layer_name"]), FMT.index(r["datatype"])
            Q[i, j], R[i, j] = f(r["Q_fisher_val_exact"]), f(r["R_exposure_unprotected_pp"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 11))
    q = np.where(Q > 0, Q, np.nan)
    im = heat(ax[0], q, [short(x) for x in d.layers], [NAME[k] for k in FMT], SEQ, LogNorm(vmin=np.nanmin(q), vmax=np.nanmax(q)), "")
    fig.colorbar(im, ax=ax[0], shrink=0.45, label="Q = sum F (w - w_hat)^2 (log)")
    ax[0].set_title("Quantization penalty Q (Fisher, validation images 0..2047)")
    r_ = np.where(R > 0, R, np.nan)
    im = heat(ax[1], r_, [short(x) for x in d.layers], [NAME[k] for k in FMT], ORG, LogNorm(vmin=np.nanmin(r_), vmax=np.nanmax(r_)), "")
    fig.colorbar(im, ax=ax[1], shrink=0.45, label="R exposure, pp (log)")
    ax[1].set_title("Unprotected fault exposure R (8 flips over the whole model)")
    save(fig, out, "07_Q_and_R_heatmaps.png", "Inputs of the optimiser: Q (Fisher) and R (exposure) per layer and datatype", "R is proportional to the layer's share of the model's weights; protection multiplies R by the measured residual rate (SECDED 0)")


def fig_q_validation(d, out):
    Qf = {(r["layer_name"], r["datatype"]): f(r["Q_pred_dCE_fisher"]) for r in d.cand if r["protection"] == "none"}
    M = {(r["layer_name"], r["format_key"]): f(r["loss_change"]) for r in d.s3}
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.5))
    for a, fm in zip(ax.flat, FMT):
        x = np.array([Qf[(l, fm)] for l in d.layers])
        y = np.array([M[(l, fm)] for l in d.layers])
        a.scatter(x, y, s=14, color=COL[fm])
        a.axhline(0, color=MUTED, lw=0.8)
        if fm in ("fp8_e4m3fn", "fp8_e5m2", "int4_pt"):
            a.set_xscale("symlog", linthresh=1e-5)
            a.set_yscale("symlog", linthresh=1e-4)
        else:
            a.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
        a.set_title(f"{NAME[fm]}   rho = {spearmanr(x, y)[0]:+.2f}", fontsize=9)
        a.set_xlabel("0.5 Q (predicted dCE)")
        a.set_ylabel("measured dCE, one layer")
    ax.flat[-1].axis("off")
    ax.flat[-1].text(0, 0.6, "rho = Spearman over the 53 layers.\nINT8 and the 16-bit formats\nare below the noise of a\n5,000-image evaluation.", fontsize=8.5, color=INK2)
    save(fig, out, "08_Q_predicted_vs_measured.png", "Fisher Q against the measured one-layer-at-a-time loss change", "Stage 3 measurement on all 5,000 validation images vs 0.5 Q from the 2,048 Fisher images (the two sets overlap; the clean check of the optimiser does not)")


def fig_selected(d, out):
    sel = d.asg["selected"]
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(53)
    for i, s in enumerate(sel):
        ax.bar(i, s["S_bytes"], color=COL[s["datatype"]], hatch="//" if s["protection"] == "scale_tmr" else "", edgecolor=SURF, width=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x, [short(s["layer_name"]) for s in sel], rotation=90, fontsize=6.5)
    ax.set_ylabel("stored bytes (log)")
    h = [Patch(color=COL[k], label=NAME[k]) for k in FMT if any(s["datatype"] == k for s in sel)] + [Patch(facecolor="#d6d5d1", label="SECDED"), Patch(facecolor="#d6d5d1", hatch="//", edgecolor=INK2, label="scale TMR")]
    ax.legend(handles=h, ncol=5, loc="upper left")
    save(fig, out, "09_selected_assignment_per_layer.png", "Final assignment (A000, Fisher Q): datatype and protection of each of the 53 layers",
         f"{d.asg['storage']['actual_stored_bytes']:,} bytes = {d.asg['storage']['avg_bits_per_weight']:.3f} bits/weight; bar colour = datatype, hatch = scale TMR (solid = SECDED)")


def fig_tradeoff(d, out):
    items = d.m["assignments"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for it in items:
        aid = it["assignment_id"]
        if aid.startswith("U_"):
            continue
        ax[0].scatter(it["storage"]["avg_bits_per_weight"], it["clean_validation"]["drop_pp_vs_fp32"], s=14, color="#b9b8b2", zorder=2)
    for it in items:
        aid = it["assignment_id"]
        if aid.startswith("U_"):
            x, y = it["storage"]["avg_bits_per_weight"], it["clean_validation"]["drop_pp_vs_fp32"]
            ax[0].scatter(x, y, s=40, marker="s", color=INK, zorder=3)
            ax[0].annotate(aid[2:], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    fin = d.e2e["A000"]
    ax[0].scatter(f(fin["avg_bits"]), f(fin["clean_drop_pp"]), s=120, marker="*", color="#e34948", zorder=4, label="final A000")
    ax[0].axhline(0.5, color=INK, ls="--", lw=1)
    ax[0].text(16.2, 0.55, "delta_clean 0.5 pp", ha="right", fontsize=8)
    ax[0].set_yscale("symlog", linthresh=0.1)
    ax[0].set_xlabel("stored bits per weight")
    ax[0].set_ylabel("measured clean accuracy drop, pp (2,952 images)")
    ax[0].set_ylim(-0.25, 20)
    ax[0].set_title("All 212 assembled models (grey = optimiser, black = uniform)", fontsize=9)
    ax[0].legend(loc="upper right")
    for it in items:
        fa = it["fault_end_to_end"].get("weights_k8")
        if not fa:
            continue
        ref = it["assignment_id"].startswith("U_")
        ax[1].scatter(it["storage"]["avg_bits_per_weight"], fa["drop_pp_mean"], s=40 if ref else 22, marker="s" if ref else "o", color=INK if ref else "#2a78d6", zorder=3)
        if ref:
            ax[1].annotate(it["assignment_id"][2:], (it["storage"]["avg_bits_per_weight"], fa["drop_pp_mean"]), xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    ax[1].scatter(f(fin["avg_bits"]), f(fin["weights_k8_drop_pp_mean"]), s=120, marker="*", color="#e34948", zorder=4)
    ax[1].axhline(0.5, color=INK, ls="--", lw=1)
    ax[1].set_yscale("symlog", linthresh=0.1)
    ax[1].set_xlabel("stored bits per weight")
    ax[1].set_ylabel("measured drop under 8 random weight-bit flips, pp (100 trials)")
    ax[1].set_ylim(-0.25, 40)
    ax[1].set_title("33 assignments with fault trials (blue) and uniform references (black)", fontsize=9)
    save(fig, out, "10_storage_accuracy_fault_tradeoff.png", "Storage, clean accuracy and fault robustness of every assembled assignment", "uniform references: names after U_ (e.g. INT8, INT4_TMR, INT8_SECDED); unprotected INT references fail the scale-word limit (not drawn here)")


def fig_pred_meas(d, out):
    it = d.m["assignments"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    x = np.array([i["predicted_layer_additive"]["sum_Q"] * 0.5 for i in it])
    y = np.array([i["clean_validation"]["delta_ce_vs_fp32"] for i in it])
    ax[0].scatter(x, y, s=12, color="#2a78d6")
    ax[0].set_xscale("symlog", linthresh=1e-4)
    ax[0].set_yscale("symlog", linthresh=1e-3)
    ax[0].set_title(f"Clean loss change (212 models)   rho = {spearmanr(x, y)[0]:+.2f}")
    ax[0].set_xlabel("predicted: 0.5 x sum Q")
    ax[0].set_ylabel("measured dCE")
    fi = [i for i in it if i["fault_end_to_end"].get("weights_k8")]
    xp = np.array([i["predicted_layer_additive"]["weight_fault_drop_pp"] for i in fi])
    yp = np.array([i["fault_end_to_end"]["weights_k8"]["drop_pp_mean"] for i in fi])
    ax[1].scatter(xp, yp, s=22, color="#eb6834")
    lim = [min(xp.min(), yp.min(), 0) - 0.05, max(xp.max(), yp.max()) * 1.05]
    ax[1].plot(lim, lim, color=MUTED, lw=1)
    ax[1].set_title(f"Weight-fault drop (33 models)   rho = {spearmanr(xp, yp)[0]:+.2f}")
    ax[1].set_xlabel("predicted: sum of layer R, pp")
    ax[1].set_ylabel("measured weights_k8 mean drop, pp")
    save(fig, out, "11_predicted_vs_measured_assignments.png", "Layer-additive estimates against the assembled models", "the optimiser predicts with sums over layers; the final decision uses the measured values")


def fig_trials(d, out):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, dist, ttl in zip(ax, ("weights_k8", "stored_k8", "scales_k1"), ("8 random weight bits", "8 random stored bits (incl. check bits, DMR copies)", "1 scale-word bit")):
        v = np.array([100 * (t["clean_accuracy"] - t["fault_accuracy"]) / 100 if t["clean_accuracy"] <= 1 else t["clean_accuracy"] - t["fault_accuracy"]
                      for t in d.trials if t["distribution"] == dist])
        a.hist(v, bins=np.arange(-0.65, 0.66, 0.1), color="#2a78d6", edgecolor=SURF)
        a.set_yscale("log")
        a.set_title(f"{ttl}\n300 trials: mean {v.mean():+.3f}, worst {v.max():.2f} pp", fontsize=9)
        a.set_xlabel("accuracy drop, pp (1,000 images)")
        a.set_ylabel("trials (log)")
    save(fig, out, "12_final_assignment_fault_trial_distributions.png", "Fault trials on the frozen assignment A000 (validation)", "all trials kept, 0 failures, byte-exact restoration after every trial")


def fig_sweep(d, out):
    rows = [r for r in d.sweep if r["mode"] == "joint" and r["normalization"] == "minmax" and f(r["B_avg_bits_per_weight"]) == 8 and r["tau_fault_pred_drop_pp"] == "" and r["status"] == "optimal"]
    rows.sort(key=lambda r: f(r["lambda"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    bottom = np.zeros(len(rows))
    for fm in FMT:
        c = np.array([sum(1 for t in r["assignment"].split(";") if TOK.get(t.split("+")[0]) == fm) for r in rows])
        ax[0].bar(range(len(rows)), c, bottom=bottom, color=COL[fm], label=NAME[fm], edgecolor=SURF)
        bottom += c
    ax[0].set_xticks(range(len(rows)), [r["lambda"] for r in rows])
    ax[0].set_xlabel("lambda (min-max, B = 8 bits, no tau_fault)")
    ax[0].set_ylabel("layers")
    ax[0].legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax[0].set_title("Datatype mix of the optimal assignment vs lambda")
    ax[1].plot([f(r["lambda"]) for r in rows], [f(r["avg_bits_per_weight"]) for r in rows], "o-", color=INK)
    ax[1].set_xscale("symlog", linthresh=0.1)
    ax[1].set_xlabel("lambda")
    ax[1].set_ylabel("stored bits per weight")
    ax[1].set_title("Average bits of the optimum")
    save(fig, out, "13_lambda_sweep_composition.png", "Effect of lambda on the Fisher-Q optimum", "full sweep (8 lambdas x 9 budgets x 5 tau x 2 normalisations x 2 modes) in normalization_lambda_sweep.csv")


def fig_vs_uniform(d, out):
    ids = ["U_FP16", "U_BF16", "U_E4M11", "U_FP8N", "U_FP8E5", "U_INT8", "U_INT4", "A000"]
    lab = ["FP16", "BF16", "E4M11", "FP8\nE4M3FN", "FP8\nE5M2", "INT8", "INT4", "A000\n(assigned)"]
    col = [COL[k] for k in FMT] + [INK]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.4))
    spec = [("avg_bits", "stored bits / weight", None), ("clean_val_accuracy", "clean validation accuracy, % (2,952 images)", (82, 93)),
            ("weights_k8_drop_pp_mean", "drop, 8 random weight bits, pp", "sym"), ("scales_k1_drop_pp_mean", "drop, 1 scale-word bit, pp (INT only)", "sym")]
    for a, (k, ttl, lim) in zip(ax, spec):
        y = [f(d.e2e[i][k]) for i in ids]
        a.bar(range(8), np.nan_to_num(y), color=col, width=0.7)
        for i, v in enumerate(y):
            a.text(i, (0 if np.isnan(v) else v) + (0.2 if lim != "sym" else 0), "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        a.set_xticks(range(8), lab, fontsize=7.5)
        a.set_title(ttl, fontsize=9)
        if lim == "sym":
            a.set_yscale("symlog", linthresh=0.1)
            a.set_ylim(0, max(np.nan_to_num(y)) * 3)
        elif lim:
            a.set_ylim(*lim)
    fp32 = f(d.m["fp32_reference"]["val_accuracy"])
    ax[1].axhline(fp32, color=INK, ls="--", lw=1)
    save(fig, out, "14_assigned_vs_uniform_unprotected.png", "Final assignment versus the uniform, unprotected models",
         "uniform = every layer in one format without protection (same fault protocol, 100 trials); scale-word bit faults apply to INT only; A000 protects all 53 layers")


def fig_val_test(d, out):
    t = d.test
    A = json.load(open(d.r / "stage_06_optimization/assignment.json"))
    TA = json.load(open(d.r / "stage_06_optimization/final_test_evaluation.json"))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    vals = [("Fisher Q (A000)", A["measured_validation_screen"]["clean_validation"] if False else None)]
    lab = ["FP32", "Fisher-Q A000", "measured-Q A000\n(comparison)"]
    test = [t["fp32_test_accuracy"], t["assembled_test_accuracy"], TA["assembled_test_accuracy"]]
    ax.bar(range(3), test, color=[MUTED, "#2a78d6", "#8a8984"], width=0.6)
    for i, v in enumerate(test):
        ax.text(i, v + 0.05, f"{v:.2f} %", ha="center", fontsize=9)
    ax.set_ylim(90, 92.5)
    ax.set_xticks(range(3), lab)
    ax.set_ylabel("official test accuracy, % (10,000 images, read once per variant)")
    save(fig, out, "15_validation_vs_test_and_variants.png", "Final test accuracy", "the assignments were frozen from validation data before the test set was read; the variant to report was NOT chosen by test accuracy")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", default="reports/mobilenetv2_cifar10")
    ap.add_argument("--out", default="reports/mobilenetv2_cifar10/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = D(a.reports)
    print("figures ->", out)
    for fn in (fig_format_overview, fig_global_screening, fig_scale_faults, fig_layer_heat, fig_quality_vs_robust, fig_protection, fig_qr_heat, fig_q_validation, fig_selected,
               fig_tradeoff, fig_pred_meas, fig_trials, fig_sweep, fig_vs_uniform, fig_val_test):
        fn(d, out)


if __name__ == "__main__":
    main()
