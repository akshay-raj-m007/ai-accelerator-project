"""Stage 5 figures (J-FAMP, datatype-only): per-layer sensitivities, Q / R / S inputs, selected assignments, predicted versus measured.

Reads only saved files of reports/scratch/jfamp/optimization (jfamp_candidate_table.csv, q_layerwise_validation.csv, fisher_exact_val_layers.json,
normalization_lambda_sweep.csv, end_to_end_assignment_metrics.json, end_to_end_fault_trials.jsonl) and writes PNGs.

  python scripts/plot_stage5_summary.py --opt-dir reports/scratch/jfamp/optimization --out plots_preview/stage5
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ style (validated reference palette, light surface)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
FORMATS = ["e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt"]
LABEL = {"e4m11_sn": "E4M11-SN", "fp16": "FP16", "fp8_e4m3fn": "FP8 E4M3FN", "int8_pt": "INT8", "int4_pt": "INT4"}
BITS = {"e4m11_sn": 16, "fp16": 16, "fp8_e4m3fn": 8, "int8_pt": 8, "int4_pt": 4}
COL = {"e4m11_sn": "#2a78d6", "fp16": "#eb6834", "fp8_e4m3fn": "#1baf7a", "int8_pt": "#6250d6", "int4_pt": "#e34948"}
SEQ_BLUE = LinearSegmentedColormap.from_list("seqblue", ["#eef5fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("seqorange", ["#fef3ec", "#f6c3a8", "#eb6834", "#b0451a", "#6e2608"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def short(layer):
    return layer.replace("layer", "L").replace(".downsample.0", ".ds").replace("conv", "c")


def title(fig, main, sub, y=0.985):
    fig.text(0.012, y, main, fontsize=15, fontweight="bold", va="top", color=INK)
    fig.text(0.012, y - 0.034, sub, fontsize=10.5, va="top", color=INK2, linespacing=1.35)


def save(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def load(opt):
    T = list(csv.DictReader(open(opt / "jfamp_candidate_table.csv", encoding="utf8")))
    layers = []
    for r in T:
        if r["layer_name"] not in layers:
            layers.append(r["layer_name"])
    tab = {(r["layer_name"], r["datatype"]): r for r in T}
    QV = list(csv.DictReader(open(opt / "q_layerwise_validation.csv", encoding="utf8")))
    fl = json.load(open(opt / "fisher_exact_val_layers.json"))["layers"]
    M = json.load(open(opt / "end_to_end_assignment_metrics.json"))
    trials = [json.loads(line) for line in open(opt / "end_to_end_fault_trials.jsonl", encoding="utf8")]
    S = list(csv.DictReader(open(opt / "normalization_lambda_sweep.csv", encoding="utf8")))
    return layers, tab, QV, fl, M, trials, S


# ------------------------------------------------------------------ 1. what Stage 5 does
def fig1(out):
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    title(fig, "Stage 5 in one picture: from measurements to a mixed-datatype network",
          "Datatype-only analysis (protection = none). Every layer of ResNet-18 gets one of five stored formats; the score tables are estimates, the assembled model is the measurement.")

    def box(x, y, w, h, head, body, fc="#ffffff", ec=GRID, hc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1.4))
        ax.text(x + 1.2, y + h - 1.6, head, fontsize=11.5, fontweight="bold", va="top", color=hc)
        ax.text(x + 1.2, y + h - 4.6, body, fontsize=9.3, va="top", color=INK2, linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))

    box(1, 36, 28, 14, "Q(l,d)  quantization penalty", "Clean accuracy layer l loses when stored\nin format d:  sum Fisher x (w - stored w)^2.\nSmall = harmless.", ec=COL["e4m11_sn"])
    box(1, 19, 28, 14, "R(l,d)  fault risk", "Accuracy drop when random bit flips hit\nlayer l stored in format d\n(Stage 4 measurements).", ec=COL["fp16"])
    box(1, 2, 28, 14, "S(l,d)  storage", "Actual stored bytes of layer l in format d\n(payload + scales). E4M11/FP16 16 bit,\nFP8/INT8 8 bit, INT4 4 bit per weight.", ec=COL["fp8_e4m3fn"])
    for y in (43, 26, 9):
        arrow(29.8, y, 34.4, 26)
    box(35, 8, 29, 36, "Optimiser (exact ILP)", "choose ONE format per layer\n\nminimise\n  sum Q_norm + lambda R_norm\n\nsubject to\n  total bytes <= B\n  (clean-accuracy and fault limits\n   are checked on the measured\n   model, not inside the ILP)\n\nsweep: 2 normalisations,\nlambda 0..10, B 4.5..16 bits\n-> 251 distinct assignments", ec=INK)
    arrow(64.8, 26, 69.4, 26)
    box(70, 31, 29, 19, "Assemble the full network", "Each layer stored in its chosen\nformat (real encoded bytes; biases\nand BatchNorm stay FP32).\nExample: early layers INT8,\nlate layers INT4.", ec=INK)
    arrow(84.5, 30.6, 84.5, 25.8)
    box(70, 8, 29, 17, "Measure end to end", "clean accuracy (2,952 images not\nused for Q), stored bytes,\nbit-flip trials in the stored bytes:\n8 flips over all weights, 50 trials,\nplus one scale-word flip (INT)", ec=INK)
    arrow(84.5, 7.6, 84.5, 4.4)
    ax.text(84.5, 2.4, "predicted vs measured differ?\nupdate layer estimates, repeat (runbook 7.2)", fontsize=9.4, color=COL["fp16"], fontweight="bold", va="center", ha="center")
    save(fig, out / "1_stage5_overview.png")


# ------------------------------------------------------------------ 2. layer sensitivity profile
def fig2(out, layers, tab, QV, fl):
    L = len(layers)
    y = np.arange(L)
    nw = np.array([int(tab[(l, "int4_pt")]["num_weights"]) for l in layers]) / 1e6
    trace = np.array([fl[l + ".weight"]["trace_per_param"] for l in layers])
    pred = np.array([float(tab[(l, "int4_pt")]["Q_pred_dCE_val"]) for l in layers])
    meas = np.array([float([r for r in QV if r["datatype"] == "int4_pt" and r["layer_name"] == l][0]["measured_dCE"]) for l in layers])
    fig, axs = plt.subplots(1, 4, figsize=(15, 8.6), sharey=True, gridspec_kw={"wspace": 0.12, "width_ratios": [1, 1, 1, 1], "top": 0.86, "bottom": 0.07, "left": 0.085, "right": 0.985})
    title(fig, "How sensitive is each layer?  (ResNet-18, 21 Conv/Linear layers, input side at the top)",
          "Early layers are small but sensitive; late layers hold most weights and are the least sensitive. Columns b-d use INT4 as the stress test, because only INT4 changes accuracy measurably.")
    blue = COL["e4m11_sn"]
    axs[0].barh(y, nw, color=blue, height=0.62)
    axs[0].set_title("a  Weights in the layer (millions)", fontsize=10.5)
    for yi, v in zip(y, nw):
        axs[0].text(v + 0.03, yi, f"{v:.3f}" if v < 0.01 else f"{v:.2f}", va="center", fontsize=8, color=INK2)
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
    axs[3].set_title("d  Measured loss change, only this\nlayer INT4 (2,952 val. images)", fontsize=10.5)
    for yi, v in zip(y, meas):
        axs[3].text(v + (0.0004 if v >= 0 else -0.0004), yi, f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2)
    axs[3].set_xlim(-0.026, 0.014)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([short(l) for l in layers], fontsize=9)
    axs[0].invert_yaxis()
    for a in axs:
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.tick_params(axis="y", length=0)
    axs[3].text(-0.0255, 1.2, "grey = quantization lowered the loss\n(over-confident logits get shrunk);\nnot 'more sensitive'", fontsize=8.4, color=INK2, va="center")
    save(fig, out / "2_layer_sensitivity_profile.png")


# ------------------------------------------------------------------ 3. Q heatmap
def annotate(ax, M, fmt, norm, cmap, thr=0.6):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            c = "white" if norm(v) > thr else INK
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.4, color=c)


def fig3(out, layers, tab):
    Q = np.array([[float(tab[(l, f)]["Q_pred_dCE_val"]) for f in FORMATS] for l in layers])
    fig, ax = plt.subplots(figsize=(9.6, 9.2))
    fig.subplots_adjust(top=0.83, bottom=0.07, left=0.16, right=0.86)
    title(fig, "Q: predicted clean-loss increase per layer and format",
          "1/2 x sum of Fisher x (weight - stored weight)^2.  Darker = this layer loses more accuracy in this format.\nThe format sets the size (INT4 >> FP8 > INT8 >> FP16 > E4M11); the layer sets the pattern inside each column.")
    norm = LogNorm(vmin=1e-11, vmax=3e-2)
    im = ax.imshow(np.maximum(Q, 1e-11), cmap=SEQ_BLUE, norm=norm, aspect="auto")
    annotate(ax, np.maximum(Q, 1e-11), lambda v: f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e+"), lambda v: norm(v), SEQ_BLUE, 0.55)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{LABEL[f]}\n{BITS[f]} bit" for f in FORMATS], fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([short(l) for l in layers], fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.88, 0.2, 0.022, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("predicted loss increase (log scale)", fontsize=9)
    save(fig, out / "3_Q_quantization_penalty_heatmap.png")


# ------------------------------------------------------------------ 4. R heatmaps
def fig4(out, layers, tab):
    R = np.array([[float(tab[(l, f)]["R_any_k8_mean_drop_pp"]) for f in FORMATS] for l in layers])
    RS = np.array([[float(tab[(l, f)]["R_scale_any_k1_mean_drop_pp"]) for f in ("int8_pt", "int4_pt")] for l in layers])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 9.6), gridspec_kw={"width_ratios": [5, 2.3], "wspace": 0.32, "top": 0.77, "bottom": 0.1, "left": 0.08, "right": 0.93})
    title(fig, "R: accuracy lost when bits flip in ONE layer  (Stage 4 screening, percentage points)",
          "Left: 8 random bit flips among the stored weights of that one layer (20 trials x 1,000 images). Right: one random bit of the layer's float32 scale (INT only; extension).\nFloat formats are badly hurt when flips land in a small early layer (exponent bits); INT weight bits are harmless; INT scale words are the INT weak spot.\nColour is clipped at the top of the bar; values below 0 are noise.")
    n1 = Normalize(vmin=0, vmax=20, clip=True)
    im1 = a1.imshow(np.clip(R, 0, None), cmap=SEQ_BLUE, norm=n1, aspect="auto")
    annotate(a1, R, lambda v: f"{v:.2f}", lambda v: n1(max(v, 0)), SEQ_BLUE, 0.55)
    a1.set_xticks(range(5))
    a1.set_xticklabels([f"{LABEL[f]}" for f in FORMATS], fontsize=9.5)
    a1.set_title("Weight bits, any bit, k = 8", fontsize=10.5, pad=30)
    n2 = Normalize(vmin=0, vmax=60, clip=True)
    im2 = a2.imshow(RS, cmap=SEQ_ORANGE, norm=n2, aspect="auto")
    annotate(a2, RS, lambda v: f"{v:.1f}", lambda v: n2(v), SEQ_ORANGE, 0.55)
    a2.set_xticks(range(2))
    a2.set_xticklabels(["INT8\nscale word", "INT4\nscale word"], fontsize=9.5)
    a2.set_title("Scale word, any of 32 bits, k = 1", fontsize=10.5, pad=30)
    for a in (a1, a2):
        a.xaxis.tick_top()
        a.set_yticks(range(len(layers)))
        a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
    a1.set_yticklabels([short(l) for l in layers], fontsize=9)
    a2.set_yticklabels([])
    for im, a, lab in ((im1, a1, "mean accuracy drop (pp), colour clipped at 20"), (im2, a2, "mean accuracy drop (pp), clipped at 60")):
        pos = a.get_position()
        cax = fig.add_axes([pos.x0, 0.06, pos.width, 0.012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=8.5)
    save(fig, out / "4_R_fault_risk_heatmaps.png")


# ------------------------------------------------------------------ 5. is Q right?
def fig5(out, layers, tab, QV, opt):
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"wspace": 0.28, "top": 0.82, "bottom": 0.12, "left": 0.07, "right": 0.975, "width_ratios": [1, 1]})
    title(fig, "Does the sensitivity score Q predict what really happens?",
          "Check on images that were NOT used to compute Q (validation 2,048-4,999). Quantize one layer (left) or all layers (right) and measure the change of the validation loss.")
    rows = [r for r in QV if r["datatype"] == "int4_pt"]
    px = np.array([float(r["predicted_dCE"]) for r in rows])
    py = np.array([float(r["measured_dCE"]) for r in rows])
    names = [r["layer_name"] for r in rows]
    a.axhline(0, color=GRID, lw=1)
    lim = 0.017
    a.plot([0, lim], [0, lim], color=MUTED, lw=1.2, ls="--")
    a.text(0.0072, 0.0135, "perfect prediction (dashed)", color=MUTED, fontsize=8.5)
    a.scatter(px, py, s=46, color=COL["int4_pt"], edgecolor=SURF, linewidth=1.2, zorder=3)
    for n, x, yv in zip(names, px, py):
        if x > 0.0028 or abs(yv) > 0.0045:
            a.annotate(short(n), (x, yv), xytext=(5, 4), textcoords="offset points", fontsize=8.2, color=INK2)
    a.set_xlim(-0.0008, lim)
    a.set_ylim(-0.021, lim)
    a.set_xlabel("predicted loss increase 1/2 Q  (INT4, one layer at a time)")
    a.set_ylabel("measured loss change")
    a.set_title("INT4, 21 layers: Spearman +0.71", fontsize=10.5)
    a.grid(color=GRID, lw=0.8)
    a.text(0.0005, -0.0205, "below 0: quantization lowered the loss (calibration effect;\nno sensitivity score can predict it)", fontsize=8, color=INK2, va="bottom")
    pred = {f: sum(float(r["predicted_dCE"]) for r in QV if r["datatype"] == f) for f in FORMATS}
    M = json.load(open(Path(opt) / "q_layerwise_validation.json"))
    meas = {f: M["per_datatype"][f]["all_layers_measured_dCE"] for f in FORMATS}
    order = list(reversed(FORMATS))
    b.set_xscale("symlog", linthresh=1e-5)
    b.axvline(0, color=GRID, lw=1)
    for i, f in enumerate(order):
        b.plot([pred[f], meas[f]], [i, i], color=GRID, lw=3, zorder=1)
        b.scatter([pred[f]], [i], s=70, facecolor=SURF, edgecolor=COL[f], linewidth=2.2, zorder=3)
        b.scatter([meas[f]], [i], s=70, color=COL[f], zorder=3)
    b.set_yticks(range(len(order)))
    b.set_yticklabels([LABEL[f] for f in order])
    b.set_xlabel("loss change when ALL layers use the format (symmetric log scale)")
    b.set_xlim(-3e-3, 0.2)
    b.set_ylim(-0.6, len(order) - 0.4)
    b.grid(axis="x", color=GRID, lw=0.8)
    b.set_title("Whole network: predicted (hollow) vs measured (filled)", fontsize=10.5)
    b.legend(handles=[Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, mew=2, ms=8, label="predicted  sum 1/2 Q"), Line2D([], [], marker="o", ls="", color=INK2, ms=8, label="measured")], loc="upper right", frameon=False, fontsize=9)
    b.text(2e-5, 0.42, "INT4: predicted 0.059, measured 0.016\n(3.7x over; before 5,000x under)", fontsize=8.6, color=INK2, va="bottom", ha="left")
    b.text(0.02, 0.62, "FP16, E4M11, INT8, FP8: measured\nchanges are within evaluation\nnoise (|change| < 1.5e-3)", transform=b.transAxes, fontsize=8.6, color=INK2, va="center", ha="left")
    save(fig, out / "5_Q_validation_predicted_vs_measured.png")


# ------------------------------------------------------------------ 6. assignments
def fig6(out, layers, tab, M, QV):
    A = {a["assignment_id"]: a for a in M["assignments"]}
    cols = ["U_INT4", "A051", "A152", "A222", "A048", "A246", "A044", "A007", "U_E4M11SN"]
    names = {"U_INT4": "uniform INT4", "U_E4M11SN": "uniform E4M11"}
    L = len(layers)
    fig = plt.figure(figsize=(15, 9.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 9], wspace=0.04, top=0.76, bottom=0.07, left=0.07, right=0.985)
    a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    title(fig, "Which format does each layer get?  Selected assignments, cheapest on the left",
          "Late layers (bottom, low sensitivity) are mostly INT4 or FP8; early, sensitive layers mostly INT8 or E4M11. The pattern is not strict: the optimiser also weighs fault risk R,\nwhich penalises float formats in small early layers (see figure 4), so some sensitive layers get INT4 (e.g. A051) and the clean accuracy loss grows accordingly.\nColumn header: average stored bits per weight / clean accuracy change vs FP32 (pp) / mean accuracy drop under 8 random weight-bit flips over the whole model (pp).")
    pred = np.array([float(tab[(l, "int4_pt")]["Q_pred_dCE_val"]) for l in layers])
    a0.barh(range(L), pred, color=COL["e4m11_sn"], height=0.62)
    a0.set_xscale("log")
    a0.set_xlim(3e-6, 6e-2)
    a0.invert_yaxis()
    a0.set_yticks(range(L))
    a0.set_yticklabels([short(l) for l in layers], fontsize=9)
    a0.set_title("sensitivity\n(1/2 Q, INT4)", fontsize=9.5, pad=70)
    a0.grid(axis="x", color=GRID, lw=0.8)
    a0.tick_params(axis="y", length=0)
    a0.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
    a1.set_xlim(-0.5, len(cols) - 0.5)
    a1.set_ylim(L - 0.5, -0.5)
    for j, aid in enumerate(cols):
        a = A[aid]
        for i, l in enumerate(layers):
            f = a["datatype_per_layer"][l]
            a1.add_patch(plt.Rectangle((j - 0.47, i - 0.46), 0.94, 0.92, fc=COL[f], ec=SURF, lw=1))
            a1.text(j, i, {"e4m11_sn": "E4M11", "fp16": "FP16", "fp8_e4m3fn": "FP8", "int8_pt": "INT8", "int4_pt": "INT4"}[f], ha="center", va="center", fontsize=8, color="white" if f != "fp8_e4m3fn" else INK, fontweight="bold")
        w = a["fault_end_to_end"]["weights_k8"]["drop_pp_mean"]
        a1.text(j, -0.85, f"{names.get(aid, aid)}\n{a['storage']['avg_bits_per_weight']:.1f} bit\nclean {-a['clean_validation']['drop_pp_vs_fp32']:+.2f} pp\nfaults {w:.2f} pp", ha="center", va="bottom", fontsize=8.8, color=INK, linespacing=1.45)
    a1.text(-0.5, -0.85, "", fontsize=8)
    a1.set_yticks([])
    a1.set_xticks([])
    for s in a1.spines.values():
        s.set_visible(False)
    a1.legend(handles=[Patch(fc=COL[f], label=f"{LABEL[f]} ({BITS[f]} bit)") for f in FORMATS], ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=9.5)
    save(fig, out / "6_selected_assignments_per_layer.png")


# ------------------------------------------------------------------ 7. trade-off
def per_assign(M):
    return {a["assignment_id"]: a for a in M["assignments"]}


def fig7(out, M):
    A = per_assign(M)
    fig, axs = plt.subplots(1, 3, figsize=(15.4, 5.9), gridspec_kw={"wspace": 0.24, "top": 0.84, "bottom": 0.13, "left": 0.055, "right": 0.985})
    title(fig, "Storage versus accuracy and fault robustness, measured on the assembled models",
          "Every dot is a full network built from an optimiser assignment. Grey: 256 assignments (clean accuracy only). Blue: 22 assignments that also got bit-flip trials. Coloured rings: uniform single-format networks.")
    bits = lambda a: a["storage"]["avg_bits_per_weight"]
    allA = list(A.values())
    sh = [a for a in allA if "weights_k8" in a["fault_end_to_end"] and not a["assignment_id"].startswith("U_")]
    uni = {a["assignment_id"]: a for a in allA if a["assignment_id"].startswith("U_")}
    ufmt = {"U_E4M11SN": "e4m11_sn", "U_FP16": "fp16", "U_FP8": "fp8_e4m3fn", "U_INT8": "int8_pt", "U_INT4": "int4_pt"}
    a, b, c = axs
    a.scatter([bits(x) for x in allA], [x["clean_validation"]["drop_pp_vs_fp32"] for x in allA], s=14, color="#c9c8c2", zorder=1)
    a.scatter([bits(x) for x in sh], [x["clean_validation"]["drop_pp_vs_fp32"] for x in sh], s=34, color=COL["e4m11_sn"], edgecolor=SURF, zorder=2)
    a.set_title("a  Clean accuracy lost vs FP32", fontsize=10.5)
    a.set_ylabel("accuracy drop (pp), lower is better")
    for k, u in uni.items():
        f = ufmt[k]
        a.scatter([bits(u)], [u["clean_validation"]["drop_pp_vs_fp32"]], s=120, facecolor="none", edgecolor=COL[f], linewidth=2.4, zorder=3)
        off = (-62, 12) if f == "e4m11_sn" else ((6, -14) if f == "fp16" else (6, 6))
        a.annotate(LABEL[f], (bits(u), u["clean_validation"]["drop_pp_vs_fp32"]), xytext=off, textcoords="offset points", fontsize=8.6, color=INK)
    b.set_yscale("symlog", linthresh=0.1)
    for x in sh:
        w = x["fault_end_to_end"]["weights_k8"]
        b.errorbar(bits(x), w["drop_pp_mean"], yerr=w["drop_pp_std"] / np.sqrt(w["n_trials"]), fmt="o", color=COL["e4m11_sn"], ms=6, elinewidth=1, capsize=0, zorder=2)
    for k, u in uni.items():
        f = ufmt[k]
        w = u["fault_end_to_end"]["weights_k8"]
        b.errorbar(bits(u), w["drop_pp_mean"], yerr=w["drop_pp_std"] / np.sqrt(w["n_trials"]), fmt="o", mfc="none", mec=COL[f], mew=2.2, ms=11, ecolor=COL[f], elinewidth=1.2, zorder=3)
        b.annotate(LABEL[f], (bits(u), w["drop_pp_mean"]), xytext=((-70, 8) if f == "e4m11_sn" else (6, 6)), textcoords="offset points", fontsize=8.6)
    b.axhline(0, color=GRID, lw=1)
    b.set_title("b  Weight-bit faults (8 flips, whole model)", fontsize=10.5)
    b.set_ylabel("mean accuracy drop (pp, symlog), +/- 1 s.e.")
    b.set_yticks([0, 0.1, 0.5, 1, 5])
    b.set_yticklabels(["0", "0.1", "0.5", "1", "5"])
    withint = [x for x in sh if "scales_k1" in x["fault_end_to_end"]]
    for x in withint:
        s = x["fault_end_to_end"]["scales_k1"]
        c.errorbar(bits(x), s["drop_pp_mean"], yerr=s["drop_pp_std"] / np.sqrt(s["n_trials"]), fmt="o", color=COL["e4m11_sn"], ms=6, elinewidth=1, zorder=2)
    for k in ("U_INT8", "U_INT4"):
        u = uni[k]
        s = u["fault_end_to_end"]["scales_k1"]
        c.errorbar(bits(u), s["drop_pp_mean"], yerr=s["drop_pp_std"] / np.sqrt(s["n_trials"]), fmt="o", mfc="none", mec=COL[ufmt[k]], mew=2.2, ms=11, ecolor=COL[ufmt[k]], zorder=3)
        c.annotate(LABEL[ufmt[k]], (bits(u), s["drop_pp_mean"]), xytext=(6, 6), textcoords="offset points", fontsize=8.6)
    c.set_title("c  One INT scale word flipped (extension)", fontsize=10.5)
    c.set_ylabel("mean accuracy drop (pp), +/- 1 s.e.")
    c.set_ylim(0, 40)
    c.text(0.02, 0.035, "only assignments that contain INT layers;\nthe 84 bytes of scale words dominate the INT risk", transform=c.transAxes, fontsize=8.4, color=INK2)
    for ax_ in axs:
        ax_.set_xlabel("average stored bits per weight")
        ax_.grid(color=GRID, lw=0.8)
        ax_.set_axisbelow(True)
        ax_.set_xlim(3.5, 16.8)
    save(fig, out / "7_storage_accuracy_fault_tradeoff.png")


# ------------------------------------------------------------------ 8. predicted vs measured (assignment level)
def fig8(out, layers, tab, M):
    A = per_assign(M)
    nw = {l: int(tab[(l, "int4_pt")]["num_weights"]) for l in layers}
    N = sum(nw.values())
    fig, axs = plt.subplots(1, 3, figsize=(15.4, 6.0), gridspec_kw={"wspace": 0.27, "top": 0.84, "bottom": 0.13, "left": 0.055, "right": 0.985})
    title(fig, "Layer-additive estimate versus measured end-to-end result  (the runbook's re-evaluation step)",
          "Left: the Q-based estimate works. Middle/right: the fault estimate is only a rough guide; the exposure-weighted version matches the middle panel in size, and the per-layer scale estimate is noise-limited.")
    a, b, c = axs
    allA = list(A.values())
    sq = np.array([x["predicted_layer_additive"]["sum_Q"] for x in allA])
    dr = np.array([x["clean_validation"]["drop_pp_vs_fp32"] for x in allA])
    a.scatter(sq, dr, s=18, color=COL["e4m11_sn"], alpha=0.8, edgecolor="none")
    a.set_xscale("log")
    a.set_xlabel("predicted: sum of Q over the 21 layers (log)")
    a.set_ylabel("measured clean accuracy drop (pp)")
    a.set_title("a  Clean accuracy (256), Spearman +0.87", fontsize=10.5)
    a.axhline(0, color=GRID, lw=1)
    sh = [x for x in allA if "weights_k8" in x["fault_end_to_end"]]
    px, py, pe, pl = [], [], [], []
    for x in sh:
        f = [x["datatype_per_layer"][l] for l in layers]
        pe.append(8 * sum(nw[l] / N * float(tab[(l, d)]["R_any_k1_mean_drop_pp"]) for l, d in zip(layers, f)))
        w = x["fault_end_to_end"]["weights_k8"]
        py.append(w["drop_pp_mean"])
        px.append(w["drop_pp_std"] / np.sqrt(w["n_trials"]))
        pl.append(x["assignment_id"])
    b.set_xscale("symlog", linthresh=0.01)
    b.set_yscale("symlog", linthresh=0.01)
    b.plot([0, 12], [0, 12], color=MUTED, ls="--", lw=1.2)
    b.errorbar(pe, py, yerr=px, fmt="o", color=COL["fp16"], ms=6, elinewidth=1)
    for x_, y_, l_ in zip(pe, py, pl):
        if l_ in ("U_FP16", "A018", "A194"):
            b.annotate(l_.replace("U_FP16", "uniform FP16"), (x_, y_), xytext=(6, -3), textcoords="offset points", fontsize=8.2, color=INK2)
    b.set_xlabel("predicted: 8 x sum_l (weight share) x (single-flip drop)  (pp)")
    b.set_ylabel("measured weight-bit fault drop (pp)")
    b.set_title("b  Weight-bit faults (27), Spearman +0.54", fontsize=10.5)
    b.set_xticks([0, 0.1, 1, 10])
    b.set_xticklabels(["0", "0.1", "1", "10"])
    b.set_yticks([0, 0.1, 1, 10])
    b.set_yticklabels(["0", "0.1", "1", "10"])
    sx, sy, se, sl = [], [], [], []
    for x in allA:
        s = x["fault_end_to_end"].get("scales_k1")
        if not s:
            continue
        f = [x["datatype_per_layer"][l] for l in layers]
        v = [float(tab[(l, d)]["R_scale_any_k1_mean_drop_pp"]) for l, d in zip(layers, f) if d in ("int8_pt", "int4_pt")]
        sx.append(np.mean(v))
        sy.append(s["drop_pp_mean"])
        se.append(s["drop_pp_std"] / np.sqrt(s["n_trials"]))
        sl.append(x["assignment_id"])
    c.plot([0, 40], [0, 40], color=MUTED, ls="--", lw=1.2)
    c.errorbar(sx, sy, yerr=se, fmt="o", color=COL["int4_pt"], ms=6, elinewidth=1)
    for x_, y_, l_ in zip(sx, sy, sl):
        if l_ in ("A164", "A202", "A063", "U_INT4"):
            c.annotate(l_, (x_, y_), xytext=(6, 3), textcoords="offset points", fontsize=8.2, color=INK2)
    c.set_xlim(0, 28)
    c.set_ylim(0, 40)
    c.set_xlabel("predicted: mean Stage 4 scale-word risk of its INT layers (pp)")
    c.set_ylabel("measured scale-word fault drop (pp)")
    c.set_title("c  Scale-word faults (20), Spearman +0.09", fontsize=10.5)
    for ax_ in axs:
        ax_.grid(color=GRID, lw=0.8)
        ax_.set_axisbelow(True)
    b.text(0.03, 0.93, "above the dashed line = estimate too low", transform=b.transAxes, fontsize=8.3, color=INK2)
    save(fig, out / "8_predicted_vs_measured_assignments.png")


# ------------------------------------------------------------------ 9. fault trial distributions
def fig9(out, trials):
    by = defaultdict(list)
    for r in trials:
        if r["assignment_id"].startswith("U_"):
            by[(r["assignment_id"], r["distribution"])].append(r["clean_accuracy"] - r["fault_accuracy"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"wspace": 0.2, "top": 0.8, "bottom": 0.12, "left": 0.06, "right": 0.985, "width_ratios": [5, 2.4]})
    fig.subplots_adjust(top=0.82)
    title(fig, "What a single bit-flip trial looks like  (uniform single-format networks, every dot = one trial)",
          "Faults are rare-event: most trials change nothing, a few collapse the network. That is why means need many trials, and why p90 / worst case are reported next to the mean.")
    ids = ["U_E4M11SN", "U_FP16", "U_FP8", "U_INT8", "U_INT4"]
    fm = {"U_E4M11SN": "e4m11_sn", "U_FP16": "fp16", "U_FP8": "fp8_e4m3fn", "U_INT8": "int8_pt", "U_INT4": "int4_pt"}
    rng = np.random.default_rng(0)
    for ax_, dist, keys, ttl in ((a, "weights_k8", ids, "8 flips among all stored weights, 50 trials each"), (b, "scales_k1", ["U_INT8", "U_INT4"], "one scale-word flip (INT only), 50 trials")):
        for i, k in enumerate(keys):
            v = np.array(by[(k, dist)])
            ax_.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=22, color=COL[fm[k]], alpha=0.75, edgecolor="none", zorder=2)
            ax_.hlines(v.mean(), i - 0.33, i + 0.33, color=INK, lw=2.4, zorder=3)
            (ax_.text(i + 0.36, v.mean(), f"mean {v.mean():.2f}", fontsize=8.6, va="center", ha="left", color=INK) if dist == "scales_k1" else ax_.text(i, max(v.mean(), 0) + 5, f"mean {v.mean():.2f}", fontsize=8.6, va="bottom", ha="center", color=INK))
        ax_.set_xticks(range(len(keys)))
        ax_.set_xticklabels([LABEL[fm[k]] for k in keys])
        ax_.set_ylabel("accuracy drop on 1,000 images (pp)")
        ax_.set_title(ttl, fontsize=10.5)
        ax_.grid(axis="y", color=GRID, lw=0.8)
        ax_.set_axisbelow(True)
        ax_.set_xlim(-0.6, len(keys) - 0.1 + 0.45)
    a.set_ylim(-3, 100)
    b.set_ylim(-3, 100)
    save(fig, out / "9_fault_trial_distributions.png")


# ------------------------------------------------------------------ 10. lambda sweep
def fig10(out, S):
    lams = sorted({float(r["lambda"]) for r in S})
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True, gridspec_kw={"wspace": 0.08, "top": 0.8, "bottom": 0.16, "left": 0.06, "right": 0.985})
    title(fig, "The lambda dial: how the optimiser's choice changes with the weight on fault risk",
          "Storage budget 8 bits per weight on average, min-max normalisation. Bars show how many of the 21 layers get each format; the number on top is the actual average bits.\nLeft: R = weight-bit faults only (INT8 wins the middle). Right: R also counts the INT scale-word risk (FP8 and E4M11 replace INT8).")
    keys = [("n_e4m11", "e4m11_sn"), ("n_fp16", "fp16"), ("n_fp8", "fp8_e4m3fn"), ("n_int8", "int8_pt"), ("n_int4", "int4_pt")]
    for ax_, risk, ttl in zip(axs, ("weights", "plus_scale"), ("R = weight-bit faults only", "R = weight-bit + INT scale-word risk")):
        rows = {float(r["lambda"]): r for r in S if r["normalization"] == "minmax" and r["risk_variant"] == risk and abs(float(r["B_avg_bits_per_weight"]) - 8) < 1e-9 and r["tau_fault_mean_layer_risk_pp"] == "" and r["eps_clean_pred_dCE"] == "" and r["status"] == "optimal"}
        bottom = np.zeros(len(lams))
        for k, f in keys:
            v = np.array([int(rows[l][k]) if l in rows else 0 for l in lams])
            ax_.bar(range(len(lams)), v, bottom=bottom, color=COL[f], edgecolor=SURF, linewidth=1.5, width=0.74, label=LABEL[f])
            for i, (bt, vv) in enumerate(zip(bottom, v)):
                if vv >= 2:
                    ax_.text(i, bt + vv / 2, str(vv), ha="center", va="center", fontsize=8.5, color="white" if f != "fp8_e4m3fn" else INK, fontweight="bold")
            bottom += v
        ax_.set_xticks(range(len(lams)))
        ax_.set_xticklabels([f"{l:g}" for l in lams])
        ax_.set_xlabel("lambda  (0 = only quantization penalty, 10 = fault risk dominates)")
        ax_.set_title(ttl, fontsize=10.5)
        ax_.grid(axis="y", color=GRID, lw=0.8)
        ax_.set_axisbelow(True)
        for i, l in enumerate(lams):
            if l in rows:
                ax_.text(i, 21.4, f"{float(rows[l]['avg_bits_per_weight']):.1f} b", ha="center", fontsize=8.4, color=INK2)
    axs[0].set_ylabel("number of layers (of 21)")
    axs[0].set_ylim(0, 23)
    axs[1].legend(handles=[Patch(fc=COL[f], label=LABEL[f]) for _, f in keys], ncol=5, loc="upper center", bbox_to_anchor=(-0.04, -0.15), frameon=False, fontsize=9.5)
    save(fig, out / "10_lambda_sweep_composition.png")


# ------------------------------------------------------------------ 11. per-layer data sheet
def fig11(out, layers, tab, QV, fl):
    L = len(layers)
    hdr = ["layer", "weights", "FP32\nMiB", "INT8\nKiB", "INT4\nKiB", "Fisher\n/weight", "1/2 Q\nINT4", "meas.\nINT4", "1/2 Q\nFP8", "R  E4M11\nk=8", "R  FP16\nk=8", "R  FP8\nk=8", "R  INT8\nk=8", "R  INT4\nk=8", "scale R\nINT8", "scale R\nINT4"]
    rows = []
    for l in layers:
        g = lambda f, c: float(tab[(l, f)][c])
        meas = float([r for r in QV if r["datatype"] == "int4_pt" and r["layer_name"] == l][0]["measured_dCE"])
        n = int(tab[(l, "int4_pt")]["num_weights"])
        rows.append([short(l), f"{n:,}", f"{4 * n / 2**20:.2f}", f"{g('int8_pt', 'S_total_bytes') / 1024:.0f}", f"{g('int4_pt', 'S_total_bytes') / 1024:.0f}", f"{fl[l + '.weight']['trace_per_param']:.1e}",
                     f"{g('int4_pt', 'Q_pred_dCE_val'):.1e}", f"{meas:+.4f}", f"{g('fp8_e4m3fn', 'Q_pred_dCE_val'):.0e}", f"{g('e4m11_sn', 'R_any_k8_mean_drop_pp'):.2f}", f"{g('fp16', 'R_any_k8_mean_drop_pp'):.2f}",
                     f"{g('fp8_e4m3fn', 'R_any_k8_mean_drop_pp'):.2f}", f"{g('int8_pt', 'R_any_k8_mean_drop_pp'):.2f}", f"{g('int4_pt', 'R_any_k8_mean_drop_pp'):.2f}", f"{g('int8_pt', 'R_scale_any_k1_mean_drop_pp'):.1f}", f"{g('int4_pt', 'R_scale_any_k1_mean_drop_pp'):.1f}"])
    fig, ax = plt.subplots(figsize=(15.5, 9.6))
    fig.subplots_adjust(top=0.86, bottom=0.03, left=0.01, right=0.99)
    ax.axis("off")
    title(fig, "Per-layer data sheet: every number the optimiser uses, for each of the 21 layers",
          "Sizes S: actual stored bytes (INT8 and INT4 include the 4-byte scale).  Fisher/weight and 1/2 Q: sensitivity inputs (validation images 0-2047).\nmeas. INT4: measured loss change when only this layer is INT4.  R k=8: mean accuracy drop (pp) for 8 random flips in this layer only (Stage 4).\nscale R: mean drop (pp) for one flipped bit of the layer's float32 scale (INT only).  Darker cell = larger value within its column.")
    tb = ax.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center", colLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.6)
    tb.scale(1, 1.55)
    colvals = {}
    for j in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        vals = np.array([float(rows[i][j]) for i in range(L)])
        colvals[j] = (np.clip(vals, 0, None), max(np.clip(vals, 0, None).max(), 1e-12))
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURF)
        if i == 0:
            cell.set_facecolor("#eceae4")
            cell.set_text_props(fontweight="bold", color=INK, fontsize=8.4)
            cell.set_height(0.062)
        elif j in colvals:
            v, mx = colvals[j]
            frac = v[i - 1] / mx
            if j in (14, 15):
                col = SEQ_ORANGE(0.08 + 0.55 * frac)
            else:
                col = SEQ_BLUE(0.04 + 0.55 * frac)
            cell.set_facecolor(col)
            cell.set_text_props(color="white" if frac > 0.75 else INK)
        if j == 0 and i > 0:
            cell.set_text_props(fontweight="bold", ha="left")
    save(fig, out / "11_per_layer_data_sheet.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opt-dir", default="reports/scratch/jfamp/optimization")
    ap.add_argument("--out", default="plots_preview/stage5")
    args = ap.parse_args()
    opt, out = Path(args.opt_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers, tab, QV, fl, M, trials, S = load(opt)
    fig1(out)
    fig2(out, layers, tab, QV, fl)
    fig3(out, layers, tab)
    fig4(out, layers, tab)
    fig5(out, layers, tab, QV, opt)
    fig6(out, layers, tab, M, QV)
    fig7(out, M)
    fig8(out, layers, tab, M)
    fig9(out, trials)
    fig10(out, S)
    fig11(out, layers, tab, QV, fl)


if __name__ == "__main__":
    main()
