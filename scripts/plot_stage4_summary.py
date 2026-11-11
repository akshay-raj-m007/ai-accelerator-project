"""Readable summary plots of Stage 2 (quality) and Stage 4 (fault screening) results. Runs locally from the CSV / npz files, no model needed.

  python scripts/plot_stage4_summary.py                     # writes PNGs to plots_preview/
  python scripts/plot_stage4_summary.py --out reports/scratch/jfamp/fault_screening/figures-faults_injected

Figures
  1_format_overview.png            accuracy after compression, SQNR, memory per weight
  2_global_screening_weights.png   accuracy after k = 1 and k = 8 random bit flips in the stored weights (all layers)
  3_global_screening_scales.png    the same for the 21 stored scale factors of INT8 / INT4
  4_layer_heatmaps.png             which layers lose the most accuracy (layer x format)
  5_quality_vs_robustness.png      SQNR (quality) against accuracy before / after faults
  6_distributions_*.png            weight / activation / logit distributions, clean vs faulted, two formats on shared axes  (needs the npz)
  7_activation_growth_*.png        largest value inside each layer, clean vs faulted                                         (needs the npz)
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

FORMATS = ["e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt"]
NAME = {"e4m11_sn": "E4M11-SN", "fp16": "FP16", "fp8_e4m3fn": "FP8 (E4M3)", "int8_pt": "INT8", "int4_pt": "INT4"}
FCOLOR = {"e4m11_sn": "#2a9d8f", "fp16": "#e76f51", "fp8_e4m3fn": "#8d6cab", "int8_pt": "#3d7ab8", "int4_pt": "#9aa5b1"}
GROUPS = [("sign", "sign bit"), ("exponent", "exponent bits"), ("fraction", "fraction / value bits"), ("all", "all bits (random)")]
SCALE_LABEL = {"sign": "scale: sign bit (1 bit)", "exponent": "scale: exponent bits (8 bits)", "fraction": "scale: fraction bits (23 bits)", "all": "scale: any of its 32 bits"}
GCOLOR = {"sign": "#4c78a8", "exponent": "#e4572e", "fraction": "#59a14f", "all": "#8c8c8c"}
CLEAN_C, FAULT_C = "#9ecae1", "#c0392b"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})


def plain(v, _=None):
    """Tick label without exponents: 0.01, 0.1, 1, 10, 1,000, 10,000."""
    if v == 0:
        return "0"
    a = abs(v)
    s = f"{a:,.0f}" if a >= 1 else f"{a:g}"
    return ("-" if v < 0 else "") + s


PLAIN = FuncFormatter(plain)


def num(v):
    """Number for legends without exponent notation."""
    return f"{v:,.0f}" if v >= 1000 else (f"{v:,.1f}" if v >= 10 else f"{v:.3g}")


def rd(path):
    return list(csv.DictReader(open(path, encoding="utf8")))


# ------------------------------------------------------------------------------------------------ figure 1
def fig_overview(qdir, out):
    acc = {r["format_key"]: r for r in rd(qdir / "clean_accuracy.csv")}
    sq = {r["format_key"]: float(r["sqnr_db"]) for r in rd(qdir / "quantization_metrics.csv") if r["module_name"] == "GLOBAL"}
    st = {r["format_key"]: r for r in rd(qdir / "storage_summary.csv")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(FORMATS))
    ax = axes[0]
    ref = float(acc["fp32"]["val_accuracy_percent"])
    vals = [float(acc[f]["val_accuracy_percent"]) for f in FORMATS]
    ax.axhline(ref, color="k", ls="--", lw=1, label=f"original FP32 model ({ref:.2f} %)")
    ax.scatter(x, vals, s=140, c=[FCOLOR[f] for f in FORMATS], zorder=3, edgecolors="k")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f} %", (i, v), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=9, bbox=dict(fc="white", ec="none", pad=1))
    ax.set_ylim(93.5, 96.3)
    ax.set_title("1. Accuracy after compressing the weights\n(5,000 validation images, higher is better)", fontsize=10)
    ax.set_ylabel("accuracy (%)  - note: axis starts at 93.5")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    s = [sq[f] for f in FORMATS]
    ax.bar(x, s, color=[FCOLOR[f] for f in FORMATS], edgecolor="k")
    for i, v in enumerate(s):
        ax.text(i, v + 1, f"{v:.1f} dB", ha="center", fontsize=9)
    ax.set_ylim(0, max(s) * 1.15)
    ax.set_title("2. How close are the compressed weights to the original?\n(SQNR in dB, higher = closer)", fontsize=10)
    ax.set_ylabel("SQNR (dB)")
    ax = axes[2]
    b = [float(st[f]["actual_bits_per_weight"]) for f in FORMATS]
    ax.bar(x, b, color=[FCOLOR[f] for f in FORMATS], edgecolor="k")
    for i, f in enumerate(FORMATS):
        ax.text(i, b[i] + 0.4, f"{b[i]:.0f} bits\n({float(st[f]['compression_ratio_weights']):.0f}x smaller)", ha="center", fontsize=9)
    ax.axhline(32, color="k", ls="--", lw=1)
    ax.text(len(FORMATS) - 0.5, 32.5, "original FP32 = 32 bits", ha="right", fontsize=8)
    ax.set_ylim(0, 37)
    ax.set_title("3. Memory used per weight\n(lower = smaller model)", fontsize=10)
    ax.set_ylabel("bits per weight")
    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels([NAME[f] for f in FORMATS])
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("What each number format costs and how good it is, before any fault", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "1_format_overview.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ figures 2 and 3
def grouped_fault_bars(ax, grows, formats, k, scope, name_layer):
    slot_w = 0.19
    for i, f in enumerate(formats):
        cleans = []
        for j, (g, _) in enumerate(GROUPS):
            r = [r for r in grows if r["format"] == f and r["scope_kind"] == scope and r["layer_name"] == name_layer and int(r["k"]) == k
                 and (r["fault_group"] == g or (g == "fraction" and r["fault_group"] == "value"))]
            x = i + (j - 1.5) * slot_w * 1.08
            if not r:
                ax.text(x, 3, "n/a", rotation=90, ha="center", va="bottom", fontsize=7, color="grey")
                continue
            r = r[0]
            cl, fa, worst = float(r["clean_accuracy"]), float(r["fault_accuracy_mean"]), float(r["clean_accuracy"]) - float(r["drop_pp_worst"])
            cleans.append(cl)
            ax.bar(x, fa, width=slot_w, color=GCOLOR[g], edgecolor="k", lw=0.5, label=(SCALE_LABEL if scope == "scales" else dict(GROUPS))[g] if i == 0 else None)
            ax.plot([x, x], [fa, worst], color="k", lw=1)
            ax.plot([x - slot_w * 0.3, x + slot_w * 0.3], [worst, worst], color="k", lw=1.6)
            d = cl - fa
            if d >= 1:
                ax.text(x, fa + 1.5, f"-{d:.0f}" if d >= 10 else f"-{d:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        if cleans:
            c = float(np.mean(cleans))
            ax.plot([i - 0.45, i + 0.45], [c, c], color="k", ls="--", lw=1.2, label="accuracy without faults" if i == 0 else None)
    ax.set_xticks(range(len(formats)))
    ax.set_xticklabels([NAME[f] for f in formats], fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_ylabel("accuracy after faults (%)  - higher is better")
    ax.grid(axis="y", alpha=0.3)


def fig_global(frows, out, scope, formats, fname, title, noun):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    nm = "GLOBAL" if scope == "weights" else "GLOBAL-SCALES"
    for ax, k in zip(axes, (1, 8)):
        grouped_fault_bars(ax, frows, formats, k, scope, nm)
        ax.set_title(f"{k} random bit{'s' if k > 1 else ''} flipped {noun}")
    h, l = axes[0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    seen["worst of the 20 trials (black tick)"] = plt.Line2D([], [], color="k", lw=1.6)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=len(seen), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005, "Each bar = mean accuracy of 20 independent fault trials (1,000 validation images each). Number above a bar = mean accuracy lost (percentage points)." + ("" if scope == "scales" else " n/a = this bit group does not exist for the format."), ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ figure 4
def fig_heatmaps(lrows, out):
    layers = []
    for r in lrows:
        if r["layer_name"] not in layers:
            layers.append(r["layer_name"])
    panels = [("all", 1, FORMATS, "1 random bit flipped\n(any bit of the weight)"), ("all", 8, FORMATS, "8 random bits flipped\n(any bit of the weight)"),
              ("exponent", 8, ["e4m11_sn", "fp16", "fp8_e4m3fn"], "8 flips, exponent bits only\n(floating-point formats only)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), gridspec_kw={"width_ratios": [5, 5, 3.4]}, sharey=True)
    vmax = 60
    im = None
    for ax, (g, k, fmts, title) in zip(axes, panels):
        M = np.full((len(layers), len(fmts)), np.nan)
        for r in lrows:
            if r["scope_kind"] == "weights" and r["fault_group"] == g and int(r["k"]) == k and r["format"] in fmts:
                M[layers.index(r["layer_name"]), fmts.index(r["format"])] = float(r["drop_pp_mean"])
        im = ax.imshow(M, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, "0" if abs(v) < 0.05 else f"{v:.1f}", ha="center", va="center", fontsize=7.5, color="white" if v > 35 else "black")
        ax.set_xticks(range(len(fmts)))
        ax.set_xticklabels([NAME[f] for f in fmts], rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        ax.spines[:].set_visible(False)
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels(layers, fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("mean accuracy lost (percentage points; colour capped at 60)")
    fig.suptitle("Which layer hurts most when its stored weights are corrupted? (faults injected into ONE layer at a time, mean of 20 trials)\n"
                 "rows = layers from the first (top) to the last (bottom); darker red = more accuracy lost (tiny negative numbers are noise)", fontsize=12, fontweight="bold")
    fig.savefig(out / "4_layer_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ figure 5
def fig_quality_vs_robust(frows, qdir, out):
    sq = {r["format_key"]: float(r["sqnr_db"]) for r in rd(qdir / "quantization_metrics.csv") if r["module_name"] == "GLOBAL"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for ax, k in zip(axes, (1, 8)):
        for f in FORMATS:
            r = [r for r in frows if r["format"] == f and r["scope_kind"] == "weights" and r["layer_name"] == "GLOBAL" and r["fault_group"] == "all" and int(r["k"]) == k][0]
            cl, fa = float(r["clean_accuracy"]), float(r["fault_accuracy_mean"])
            ax.annotate("", xy=(sq[f], fa), xytext=(sq[f], cl), arrowprops=dict(arrowstyle="->", color=FCOLOR[f], lw=2))
            ax.scatter([sq[f]], [cl], s=90, facecolors="white", edgecolors=FCOLOR[f], lw=2, zorder=3)
            ax.scatter([sq[f]], [fa], s=90, color=FCOLOR[f], edgecolors="k", zorder=4)
            dy, ha = {"e4m11_sn": (-36, "right"), "fp16": (16, "left")}.get(f, (-36, "center"))
            ax.annotate(f"{NAME[f]}\n{cl:.1f} -> {fa:.1f} %", (sq[f], max(cl, fa) if dy > 0 else min(cl, fa)), textcoords="offset points",
                        xytext=(0, dy), ha=ha, fontsize=8.5)
        ax.set_ylim(80, 100)
        ax.set_xlim(0, max(sq[f] for f in FORMATS) + 12)
        ax.set_xlabel("SQNR of the compressed weights (dB): higher = closer to the original")
        ax.set_title(f"{k} random bit{'s' if k > 1 else ''} flipped in the stored weights")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy (%)  - note: axis starts at 80")
    axes[0].scatter([], [], s=90, facecolors="white", edgecolors="k", lw=2, label="hollow = before faults")
    axes[0].scatter([], [], s=90, color="grey", edgecolors="k", label="filled = after faults (mean of 20 trials)")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Compression quality (SQNR) vs. robustness: the arrow is the accuracy lost to random bit flips (any bit, all layers)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "5_quality_vs_robustness.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ figures 6 and 7 (npz)
def _plain_axes(ax, edges_lim, lin):
    ax.set_xscale("symlog", linthresh=lin, linscale=0.6)
    ax.set_xlim(-edges_lim, edges_lim)
    ticks = [t for m in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4) for t in (-m, m) if t != 0 and abs(t) <= edges_lim * 1.001] + [0]
    ax.set_xticks(sorted(ticks))
    ax.xaxis.set_major_formatter(PLAIN)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.tick_params(axis="x", labelsize=7.5, rotation=60)


def fig_distributions(z, out):
    fmts = [str(f) for f in z["formats"]]
    edges = z["edges"]
    lin = float(edges[edges > 0].min())
    lname, k, group, n_img = str(z["faulted_layer"]), int(z["k"]), str(z["group"]), int(z["n_images"])
    cols = [("w", f"Weights of layer {lname}"), ("a", f"Output of layer {lname}\n(for {n_img:,} images)"), ("l", f"Final network output (logits)\n({n_img:,} images x 10 classes)")]
    lim = {c: 0.0 for c, _ in cols}
    for f in fmts:
        for c, _ in cols:
            lim[c] = max(lim[c], float(z[f"{f}__{c}_fault_stats"][3 if c != "l" else 3]), float(z[f"{f}__{c}_clean_stats"][3]))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.4), sharey="col")
    mid = 0.5 * (edges[:-1] + edges[1:])
    for r, f in enumerate(fmts):
        acc_c, acc_f = float(z[f"{f}__clean_accuracy"]), float(z[f"{f}__fault_accuracy"])
        for c, (key, title) in enumerate(cols):
            ax = axes[r, c]
            cc, cf = z[f"{f}__{key}_clean_counts"].astype(float), z[f"{f}__{key}_fault_counts"].astype(float)
            sc, sf = z[f"{f}__{key}_clean_stats"], z[f"{f}__{key}_fault_stats"]
            ax.stairs(np.where(cc > 0, cc, np.nan), edges, fill=True, color=CLEAN_C, alpha=0.8, label=f"clean: largest |value| = {num(sc[3])}")
            ax.stairs(np.where(cf > 0, cf, np.nan), edges, color=FAULT_C, lw=1.4, label=f"after faults: largest |value| = {num(sf[3])}")
            _plain_axes(ax, lim[key] * 2.5, lin)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(PLAIN)
            ax.set_ylim(0.6, None)
            if key == "w":
                old, new = z[f"{f}__flipped_old"], z[f"{f}__flipped_new"]
                for o, n in zip(old, new):
                    ax.annotate("", xy=(n, 0.8), xytext=(o, 0.8), arrowprops=dict(arrowstyle="->", color="k", lw=0.7))
                ax.scatter(new, [0.8] * len(new), marker="x", color="darkred", s=30, zorder=5, label=f"the {len(new)} corrupted weights (arrow: old -> new)")
            ax.set_xlabel("value", fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"{NAME.get(f, f)}\naccuracy {acc_c:.1f} % -> {acc_f:.1f} %\n\nnumber of values (log scale)", fontsize=9.5,
                              color=FCOLOR.get(f, "k"), fontweight="bold")
            else:
                ax.set_ylabel("number of values", fontsize=9)
            if r == 0:
                ax.set_title(title, fontsize=10.5)
            nf = int(sf[4]) if key != "w" else 0
            ax.legend(fontsize=7.5, loc="upper left" if key != "w" else "upper right", title=f"non-finite after faults: {nf}" if key != "w" else None, title_fontsize=7)
            ax.grid(alpha=0.25)
    fig.suptitle(f"What {k} flipped {group} bits in {lname} do to the numbers inside the network: {NAME.get(fmts[0], fmts[0])} vs {NAME.get(fmts[1], fmts[1])}\n"
                 "same axes in each column so the two formats can be compared directly; blue = clean model, red line = the same model after the faults\n"
                 "(worst of 20 recorded trials; the horizontal axis is stretched around 0 so small and very large values are both visible)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / f"6_distributions_{fmts[0]}_vs_{fmts[1]}.png", dpi=150)
    plt.close(fig)

    # figure 7: largest value in each layer
    names = [str(n) for n in z["layer_names"]]
    li = names.index(lname)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    x = np.arange(len(names))
    base = np.maximum(z[f"{fmts[0]}__maxabs_clean"], 1e-6)
    ax.plot(x, base, "o-", color="k", lw=2, label="clean model")
    for f in fmts:
        ax.plot(x, np.maximum(z[f"{f}__maxabs_fault"], 1e-6), "s-", color=FCOLOR.get(f, "r"), lw=2, label=f"{NAME.get(f, f)} after the faults")
    ax.axvline(li, color="grey", ls="--")
    ax.text(li + 0.15, ax.get_ylim()[0] * 1.4, f"faults injected here ({lname})", rotation=90, va="bottom", fontsize=8.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(PLAIN)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("largest value produced by the layer")
    ax.set_title("How far the damage spreads: the largest value inside each layer, clean vs after the same kind of fault\n"
                 "(a healthy network keeps its values small; the corrupted network produces much larger values in every layer)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"7_activation_growth_{fmts[0]}_vs_{fmts[1]}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fault-dir", default="reports/scratch/jfamp/fault_screening")
    ap.add_argument("--quant-dir", default="reports/scratch/jfamp/quantization")
    ap.add_argument("--out", default="plots_preview")
    ap.add_argument("--npz", default=None, help="default <fault-dir>/fault_distribution_data.npz (figures 6 and 7 are skipped if missing)")
    a = ap.parse_args()
    fd, qd, out = Path(a.fault_dir), Path(a.quant_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grows = rd(fd / "global_fault_summary.csv")
    lrows = rd(fd / "fault_summary.csv")
    fig_overview(qd, out)
    fig_global(grows, out, "weights", FORMATS, "2_global_screening_weights.png",
               "Global screening: accuracy of the model after random bit flips in the stored weights (all 11.2 million weights)", "among the 11.2 million stored weights")
    fig_global(grows, out, "scales", ["int8_pt", "int4_pt"], "3_global_screening_scales.png",
               "Extra test for the integer formats: bit flips in the 21 stored scale factors (one float32 number per layer; the integer weights are not touched)", "among the 21 stored scale factors")
    fig_heatmaps(lrows, out)
    fig_quality_vs_robust(grows, qd, out)
    npz = Path(a.npz) if a.npz else fd / "fault_distribution_data.npz"
    if npz.exists():
        fig_distributions(dict(np.load(npz, allow_pickle=False)), out)
    else:
        print("no", npz, "-> figures 6 and 7 skipped")
    print("written:", *sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
