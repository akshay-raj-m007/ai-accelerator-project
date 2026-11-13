"""Stage 6 (runbook Section 8): final analysis and presentation package, rebuilt from saved raw records only.

Reads (all under --base, default reports/<run>/jfamp): baseline/, layer_profile/, quantization/, fault_injection/, fault_screening/, optimization/.
Writes (also under --base):
  final_tables/<tag>_*.csv      required comparisons (runbook 8.1)
  figures/<tag>_fig1..fig10_*.png   recommended figures (runbook 8.2) plus the alignment, protection and validation-vs-test figures
  final_summary.md              architecture, checkpoint hash, split, formats, scope, fault groups, k, trials, sample counts, exclusions, deviations, limitations
  final_manifest.json           SHA-256 of every input and output file

No GPU and no model are needed. The test set is not read here: test numbers come from the saved files of Stages 2 and 5.

  python scripts/stage6_final_package.py --base reports/scratch/jfamp --model-tag resnet18_cifar10
For another model: run the pipeline with another --run-name, then repeat with that --base and a new --model-tag.
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from scipy import stats

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
FORMATS = ["e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt"]
ALL_FORMATS = ["fp32"] + FORMATS + ["e4m11_ftz"]
LABEL = {"fp32": "FP32", "e4m11_sn": "E4M11-SN", "fp16": "FP16", "fp8_e4m3fn": "FP8 E4M3FN", "int8_pt": "INT8", "int4_pt": "INT4", "e4m11_ftz": "E4M11-FTZ"}
COL = {"fp32": "#52514e", "e4m11_sn": "#2a78d6", "fp16": "#eb6834", "fp8_e4m3fn": "#1baf7a", "int8_pt": "#6250d6", "int4_pt": "#e34948", "e4m11_ftz": "#8a8984"}
BITS = {"e4m11_sn": 16, "fp16": 16, "fp8_e4m3fn": 8, "int8_pt": 8, "int4_pt": 4}
TOKEN_BITS = {"E4M11": 16, "FP16": 16, "FP8": 8, "INT8": 8, "INT4": 4, "INT8+T": 8, "INT4+T": 4}
SEQ = LinearSegmentedColormap.from_list("seq", ["#fef6f0", "#f9cdb4", "#eb6834", "#b0451a", "#5e2108"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})
TAG = "model"


# ------------------------------------------------------------------ helpers
def rd(path):
    with open(path, encoding="utf8", newline="") as fh:
        return list(csv.DictReader(fh))


def fl(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def wcsv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def short(layer):
    return layer.replace("layer", "L").replace(".downsample.0", ".ds").replace("conv", "c")


def title(fig, main, sub, y=0.985):
    fig.text(0.012, y, main, fontsize=15, fontweight="bold", va="top", color=INK)
    fig.text(0.012, y - 0.034, sub, fontsize=10.5, va="top", color=INK2, linespacing=1.35)


def save(fig, figdir, name):
    path = figdir / f"{TAG}_{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)
    return path


def pareto(points):
    """Indices of non-dominated rows (all coordinates minimised)."""
    P = np.asarray(points, float)
    keep = []
    for i in range(len(P)):
        dominated = any(np.all(P[j] <= P[i]) and np.any(P[j] < P[i]) for j in range(len(P)) if j != i)
        if not dominated:
            keep.append(i)
    return keep


# ------------------------------------------------------------------ data
def load(base):
    D = {"base": base}
    D["clean"] = {r["format_key"]: r for r in rd(base / "quantization/clean_accuracy.csv")}
    D["storage"] = {r["format_key"]: r for r in rd(base / "quantization/storage_summary.csv")}
    D["qm"] = rd(base / "quantization/quantization_metrics.csv")
    D["gfs"] = rd(base / "fault_screening/global_fault_summary.csv")
    D["lfh"] = rd(base / "fault_screening/layer_fault_heatmap.csv")
    D["inv"] = rd(base / "layer_profile/layer_inventory.csv")
    D["e2e"] = rd(base / "optimization/end_to_end_assignments.csv")
    D["assignment"] = json.load(open(base / "optimization/assignment.json", encoding="utf8"))
    D["final_test"] = json.load(open(base / "optimization/final_test_evaluation.json", encoding="utf8"))
    D["e2e_metrics"] = json.load(open(base / "optimization/end_to_end_assignment_metrics.json", encoding="utf8"))
    D["baseline_manifest"] = json.load(open(base / "baseline/baseline_manifest.json", encoding="utf8"))
    D["quant_manifest"] = json.load(open(base / "quantization/quantizer_manifest.json", encoding="utf8"))
    D["fault_protocol"] = json.load(open(base / "fault_injection/fault_protocol.json", encoding="utf8"))
    D["campaign"] = json.load(open(base / "fault_screening/fault_campaign_manifest.json", encoding="utf8"))
    D["layers"] = [r["module_name"] for r in D["inv"]]
    D["n_w"] = {r["module_name"]: int(r["num_weights"]) for r in D["inv"]}
    D["total_w"] = sum(D["n_w"].values())
    D["measured"] = [r for r in D["e2e"] if r["weights_k8_n_trials"] not in ("", "0")]
    D["by_id"] = {r["assignment_id"]: r for r in D["e2e"]}
    return D


def assignment_storage_parts(row, D):
    """payload / scale / protection bytes of an assembled assignment, from its layer tokens (cross-checked with the measured stored bytes)."""
    toks = row["assignment"].split(";")
    payload = scale = prot = 0
    theo = 0.0
    for tok, layer in zip(toks, D["layers"]):
        n = D["n_w"][layer]
        bits = TOKEN_BITS[tok]
        payload += math.ceil(n * bits / 8)
        theo += n * bits / 8
        if tok.startswith("INT"):
            scale += 4
        if tok.endswith("+T"):
            prot += 8
    total = fl(row["storage_bytes"])
    if abs(payload + scale + prot - total) > 0.5:
        print(f"warning: {row['assignment_id']}: parts {payload + scale + prot} differ from stored bytes {total}")
    return payload, scale, prot, theo


# ------------------------------------------------------------------ tables (8.1)
def table_formats(D):
    rows = []
    for f in ALL_FORMATS:
        c, s = D["clean"][f], D["storage"][f]
        rows.append({"format": f, "variant": c["variant"], "role": c["role"],
                     "val_images": int(fl(c["val_total"])), "val_accuracy_pct": fl(c["val_accuracy_percent"]), "val_drop_pp_vs_fp32": fl(c["val_drop_pp"]),
                     "val_cross_entropy": fl(c["val_cross_entropy"]), "val_delta_ce_vs_fp32": fl(c["val_delta_ce"]),
                     "test_images": int(fl(c["test_total"])), "test_accuracy_pct": fl(c["test_accuracy_percent"]), "test_drop_pp_vs_fp32": fl(c["test_drop_pp"]),
                     "test_cross_entropy": fl(c["test_cross_entropy"]), "test_delta_ce_vs_fp32": fl(c["test_delta_ce"]),
                     "stored_bytes_weights": int(fl(s["total_bytes"])), "payload_bytes": int(fl(s["payload_bytes"])), "scale_bytes": int(fl(s["scale_bytes"])),
                     "theoretical_payload_bits": int(fl(s["theoretical_bits"])), "bits_per_weight_actual": fl(s["actual_bits_per_weight"]),
                     "compression_ratio_weights": fl(s["compression_ratio_weights"]), "compression_ratio_payload_only": fl(s["compression_ratio_payload_only"]),
                     "compression_ratio_full_model": fl(s["compression_ratio_full_model"])})
    return rows


def table_fault(D):
    keep = ["format", "scope_kind", "fault_group", "k", "n_trials", "sample_count", "clean_accuracy", "fault_accuracy_mean", "drop_pp_mean", "drop_pp_median", "drop_pp_p90",
            "drop_pp_worst", "drop_pp_sem", "loss_increase_mean", "loss_increase_median", "loss_increase_p90", "loss_increase_worst", "failures", "failure_rate",
            "fail_weights_nonfinite", "fail_activations_nonfinite", "fail_logits_nonfinite", "fail_loss_nonfinite"]
    return [{k: r[k] for k in keep} for r in D["gfs"]]


def layer_metrics(D):
    """Per layer x format: size, weight statistics, SQNR, measured risk (any bit, k = 1 and k = 8, in ONE layer)."""
    inv = {r["module_name"]: r for r in D["inv"]}
    sq = {(r["format_key"], r["module_name"]): fl(r["sqnr_db"]) for r in D["qm"] if r["layer_id"] != ""}
    rows = []
    for i, h in enumerate(D["lfh"]):
        name = h["layer_name"]
        for f in FORMATS:
            rows.append({"layer_id": i, "layer_name": name, "format": f, "num_weights": int(inv[name]["num_weights"]), "weight_std": fl(inv[name]["weight_std"]),
                         "weight_abs_max": fl(inv[name]["weight_abs_max"]), "abs_max_over_std": fl(inv[name]["weight_abs_max"]) / fl(inv[name]["weight_std"]),
                         "sqnr_db": sq[(f, name)], "risk_k1_mean_drop_pp": fl(h[f"{f}|weights|all|k1|mean_drop_pp"]), "risk_k1_p90_drop_pp": fl(h[f"{f}|weights|all|k1|p90_drop_pp"]),
                         "risk_k8_mean_drop_pp": fl(h[f"{f}|weights|all|k8|mean_drop_pp"]), "risk_k8_p90_drop_pp": fl(h[f"{f}|weights|all|k8|p90_drop_pp"])})
    for f in FORMATS:                                                  # flag the three highest-risk layers per format and k = 8
        sub = sorted([r for r in rows if r["format"] == f], key=lambda r: -r["risk_k8_mean_drop_pp"])
        top = {r["layer_name"] for r in sub[:3]}
        for r in rows:
            if r["format"] == f:
                r["top3_risk_k8"] = r["layer_name"] in top
    return rows


PREDICTORS = [("num_weights", "layer size (weights)"), ("weight_std", "weight std"), ("weight_abs_max", "max |w|"), ("sqnr_db", "SQNR of the format (dB)"), ("layer_id", "depth (layer index)")]


def table_alignment(LM):
    rows = []
    for f in FORMATS:
        sub = [r for r in LM if r["format"] == f]
        for kk in ("k1", "k8"):
            y = [r[f"risk_{kk}_mean_drop_pp"] for r in sub]
            for key, lab in PREDICTORS:
                x = [r[key] for r in sub]
                if len(set(y)) < 2 or len(set(x)) < 2:
                    rho, p = float("nan"), float("nan")
                else:
                    rho, p = stats.spearmanr(x, y)
                rows.append({"format": f, "k": kk[1:], "predictor": key, "predictor_label": lab, "n_layers": len(sub), "spearman_rho": float(rho), "p_value": float(p),
                             "significant_5pct": bool(p < 0.05)})
    return rows


def table_protection(D):
    rows = []
    for r in D["measured"]:
        if "+T" not in r["assignment"]:
            continue
        key = r["assignment"].replace("+T", "")
        u = next((x for x in D["measured"] if x["assignment"] == key and x["assignment_id"] != r["assignment_id"]), None)
        if u is None:
            continue
        rows.append({"unprotected_id": u["assignment_id"], "protected_id": r["assignment_id"], "avg_bits_unprotected": fl(u["avg_bits"]), "avg_bits_protected": fl(r["avg_bits"]),
                     "stored_bytes_unprotected": int(fl(u["storage_bytes"])), "stored_bytes_protected": int(fl(r["storage_bytes"])),
                     "protection_overhead_bytes": int(fl(r["storage_bytes"]) - fl(u["storage_bytes"])), "protected_layers": r["n_protected_layers"],
                     "clean_drop_pp_unprotected": fl(u["clean_drop_pp"]), "clean_drop_pp_protected": fl(r["clean_drop_pp"]),
                     "weights_k8_mean_drop_pp_unprotected": fl(u["weights_k8_drop_pp_mean"]), "weights_k8_mean_drop_pp_protected": fl(r["weights_k8_drop_pp_mean"]),
                     "scales_k1_mean_drop_pp_unprotected": fl(u["scales_k1_drop_pp_mean"]), "scales_k1_mean_drop_pp_protected": fl(r["scales_k1_drop_pp_mean"]),
                     "scales_k1_worst_drop_pp_unprotected": fl(u["scales_k1_drop_pp_worst"]), "scales_k1_worst_drop_pp_protected": fl(r["scales_k1_drop_pp_worst"]),
                     "trials_per_distribution": int(fl(r["weights_k8_n_trials"])), "fault_images": int(fl(r["weights_k8_sample_count"]))})
    return rows


def table_tradeoff(D):
    rows = []
    for r in D["e2e"]:
        meas = r["weights_k8_n_trials"] not in ("", "0")
        has_scale = r["scales_k1_n_trials"] not in ("", "0")
        rows.append({"assignment_id": r["assignment_id"], "modes": r["modes"], "avg_bits": fl(r["avg_bits"]), "stored_bytes": int(fl(r["storage_bytes"])),
                     "protected_layers": r["n_protected_layers"], "clean_drop_pp": fl(r["clean_drop_pp"]), "clean_images": int(fl(D["e2e_metrics"]["fp32_reference"]["val_samples"])),
                     "fault_measured": meas, "weights_k8_mean_drop_pp": fl(r["weights_k8_drop_pp_mean"]), "weights_k8_p90_drop_pp": fl(r["weights_k8_drop_pp_p90"]),
                     "weights_k8_worst_drop_pp": fl(r["weights_k8_drop_pp_worst"]),
                     "scales_k1_mean_drop_pp": fl(r["scales_k1_drop_pp_mean"]) if has_scale else (0.0 if meas else float("nan")),
                     "scales_k1_worst_drop_pp": fl(r["scales_k1_drop_pp_worst"]) if has_scale else (0.0 if meas else float("nan")),
                     "fault_trials_per_distribution": int(fl(r["weights_k8_n_trials"])) if meas else 0})
    c = pareto([[r["avg_bits"], r["clean_drop_pp"]] for r in rows])
    for i, r in enumerate(rows):
        r["pareto_bits_vs_clean_drop"] = i in c
    midx = [i for i, r in enumerate(rows) if r["fault_measured"]]
    c4 = pareto([[rows[i]["avg_bits"], rows[i]["clean_drop_pp"], rows[i]["weights_k8_mean_drop_pp"], rows[i]["scales_k1_mean_drop_pp"]] for i in midx])
    front = {midx[j] for j in c4}
    for i, r in enumerate(rows):
        r["pareto_measured_4d"] = i in front
        r["frozen"] = r["assignment_id"] == D["assignment"]["assignment_id"]
    return rows


def table_val_test(D):
    rows = []
    ft, A = D["final_test"], D["assignment"]
    for f in ALL_FORMATS:
        c = D["clean"][f]
        rows.append({"item": LABEL[f] + " (uniform)", "kind": "uniform format", "val_images": int(fl(c["val_total"])), "val_drop_pp": fl(c["val_drop_pp"]),
                     "test_images": int(fl(c["test_total"])), "test_drop_pp": fl(c["test_drop_pp"]), "used_for_selection": "no (reported only)"})
    mv = A["measured_validation_screen"]["clean_validation"]
    rows.append({"item": f"{A['assignment_id']} (frozen assignment)", "kind": "frozen mixed assignment", "val_images": mv["samples"], "val_drop_pp": mv["drop_pp_vs_fp32"],
                 "test_images": ft["test_samples"], "test_drop_pp": ft["test_clean_drop_pp_vs_fp32"], "used_for_selection": "validation only; test read once after freezing"})
    return rows


# ------------------------------------------------------------------ figures (8.2)
def fig1(D, figdir):
    T = table_formats(D)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.1, right=0.985, top=0.80, bottom=0.12, wspace=0.28)
    ys = np.arange(len(T))[::-1]
    for y, r in zip(ys, T):
        f = r["format"]
        for off, (k, lab, mk) in zip((0.16, -0.16), (("val", "validation", "o"), ("test", "test", "s"))):
            a1.scatter(r[f"{k}_accuracy_pct"], y + off, s=60, marker=mk, color=COL[f], zorder=3)
    a1.set_yticks(ys)
    a1.set_yticklabels([LABEL[r["format"]] for r in T])
    lo = 93.0
    a1.set_xlim(lo, 96.0)
    ftz = [r for r in T if r["format"] == "e4m11_ftz"][0]
    a1.annotate(f"E4M11-FTZ collapses: {ftz['val_accuracy_pct']:.1f} % (val) / {ftz['test_accuracy_pct']:.1f} % (test), off scale", (lo, ys[-1]), xytext=(lo + 0.05, ys[-1] - 0.04),
                fontsize=9.5, color=INK2, va="center")
    a1.axvline(T[0]["test_accuracy_pct"], color=MUTED, lw=0.8, ls=":")
    a1.set_xlabel("top-1 accuracy (%)")
    a1.set_title("Accuracy", pad=8)
    a1.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label=f"validation ({T[0]['val_images']:,} images)"),
                       Line2D([], [], marker="s", ls="", color=MUTED, label=f"test ({T[0]['test_images']:,} images)")], loc="upper right", frameon=False)
    cap = 2.6
    for y, r in zip(ys, T):
        for off, k in zip((0.16, -0.16), ("val", "test")):
            d = r[f"{k}_drop_pp_vs_fp32"]
            w = min(max(d, -0.3), cap)
            a2.barh(y + off, w, height=0.28, color=COL[r["format"]], alpha=1.0 if k == "val" else 0.55)
            txt = f"{d:+.2f}" if abs(d) < 50 else f"{d:+.1f} (clipped)"
            a2.text(w + (0.04 if w >= 0 else -0.04), y + off, txt, va="center", ha="left" if w >= 0 else "right", fontsize=8.5, color=INK2)
    a2.set_yticks(ys)
    a2.set_yticklabels([])
    a2.set_xlim(-0.5, cap + 0.55)
    a2.axvline(0, color=INK2, lw=0.8)
    a2.set_xlabel("accuracy drop vs FP32 on the same images (pp); negative = higher than FP32")
    a2.set_title("Drop (pp), pale = test", pad=8)
    title(fig, "Fig. 1 - Baseline and clean accuracy: FP32 versus every candidate format",
          "Weight-only quantization of the 21 Conv/Linear tensors. Differences of 0.1 pp are 5 images of 5,000 or 10 images of 10,000, i.e. noise; INT4 (about -1.2 pp) and\n"
          "E4M11-FTZ (collapse) are the only clear losses. FP8 and INT8 differ from FP32 by less than 0.2 pp.")
    return save(fig, figdir, "fig1_baseline_clean_accuracy")


def fig2(D, figdir):
    T = {r["format"]: r for r in table_formats(D)}
    items = [(LABEL[f], fl(T[f]["payload_bytes"]), fl(T[f]["scale_bytes"]), 0.0, fl(T[f]["theoretical_payload_bits"]) / 8, fl(T[f]["compression_ratio_weights"]), fl(T[f]["bits_per_weight_actual"]))
             for f in ["fp32", "fp16", "e4m11_sn", "fp8_e4m3fn", "int8_pt", "int4_pt"]]
    A = D["assignment"]["assignment_id"]
    pick = [("U_INT8_TMR", "INT8 + scale_tmr (uniform)"), (A, f"{A} frozen (mixed + scale_tmr)")]
    for aid in ("A156",):
        if aid in D["by_id"] and aid != A:
            pick.append((aid, f"{aid} (mixed, 5-bit budget variant)"))
    fp32 = fl(T["fp32"]["stored_bytes_weights"])
    for aid, lab in pick:
        r = D["by_id"][aid]
        pay, sc, pr, th = assignment_storage_parts(r, D)
        items.append((lab, pay, sc, pr, th, fp32 / fl(r["storage_bytes"]), fl(r["avg_bits"])))
    fig, ax = plt.subplots(figsize=(14, 6.4))
    fig.subplots_adjust(left=0.27, right=0.985, top=0.80, bottom=0.12)
    ys = np.arange(len(items))[::-1]
    mib = 1 << 20
    for y, (lab, pay, sc, pr, th, cr, bpw) in zip(ys, items):
        ax.barh(y, pay / mib, height=0.58, color="#2a78d6", label="payload (weight words)" if y == ys[0] else None)
        ax.barh(y, sc / mib, left=pay / mib, height=0.58, color="#eb6834", label="scale words (float32)" if y == ys[0] else None)
        ax.barh(y, pr / mib, left=(pay + sc) / mib, height=0.58, color="#1baf7a", label="protection overhead (scale_tmr)" if y == ys[0] else None)
        ax.plot([th / mib] * 2, [y - 0.36, y + 0.36], color=INK, lw=1.4)
        tot = (pay + sc + pr) / mib
        extra = f"\nscale words {int(sc):,} B, protection copies {int(pr):,} B" if (sc or pr) else ""
        ax.text(tot + 0.4, y, f"{tot:.2f} MiB   {cr:.2f}x   {bpw:.2f} bit/weight{extra}", va="center", fontsize=9.5, color=INK2, linespacing=1.3)
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items])
    ax.set_xlim(0, max((i[1] + i[2] + i[3]) for i in items) / mib * 1.45)
    ax.set_xlabel("stored size of the 21 quantized weight tensors (MiB)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", visible=False)
    title(fig, "Fig. 2 - Actual storage, including metadata",
          "Bars are the real stored bytes (payload + float32 scales + protection copies); the black tick is the theoretical payload n x bits / 8. Compression ratio is against the FP32\n"
          "weights (42.59 MiB). Scale words and protection copies are only a few hundred bytes (too thin to see; exact bytes are printed). Biases and BatchNorm (0.07 MiB) are not included.")
    return save(fig, figdir, "fig2_storage")


def fig3(D, figdir):
    G = {r["format_key"]: fl(r["sqnr_db"]) for r in D["qm"] if r["layer_id"] == ""}
    per = {f: {r["module_name"]: fl(r["sqnr_db"]) for r in D["qm"] if r["format_key"] == f and r["layer_id"] != ""} for f in FORMATS + ["e4m11_ftz"]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.3), gridspec_kw={"width_ratios": [1, 2.1]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.2, wspace=0.18)
    fm = FORMATS + ["e4m11_ftz"]
    a1.bar(range(len(fm)), [G[f] for f in fm], color=[COL[f] for f in fm], width=0.62)
    for i, f in enumerate(fm):
        a1.text(i, G[f] + 1.2, f"{G[f]:.1f}", ha="center", fontsize=9.5)
    a1.set_xticks(range(len(fm)))
    a1.set_xticklabels([LABEL[f] for f in fm], rotation=35, ha="right")
    a1.set_ylabel("SQNR (dB), all 21 tensors pooled")
    a1.set_title("Global", pad=8)
    a1.set_ylim(0, max(G.values()) * 1.15)
    a1.grid(axis="x", visible=False)
    xs = np.arange(len(D["layers"]))
    for f in fm:
        a2.plot(xs, [per[f][n] for n in D["layers"]], marker="o", ms=4, lw=1.6, color=COL[f], label=LABEL[f])
    a2.set_xticks(xs)
    a2.set_xticklabels([short(n) for n in D["layers"]], rotation=60, ha="right", fontsize=9)
    a2.set_ylabel("SQNR (dB)")
    a2.set_title("Per layer", pad=8)
    a2.legend(ncol=6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    title(fig, "Fig. 3 - Quantization quality (SQNR = 10 log10 sum w^2 / sum (w - w^)^2)",
          "Higher is better. FP16 and E4M11-SN are near-lossless, INT8 and FP8 are 20-35 dB, INT4 about 8 dB. E4M11-FTZ (comparison only) flushes small weights to zero\n"
          "and collapses, which is why it is not a candidate. Quality is a weight-space measure; it ranks formats but does not predict accuracy loss by itself (see Fig. 7).")
    return save(fig, figdir, "fig3_sqnr")


def fig4(D, figdir):
    G = D["gfs"]
    groups_w = {f: [g for g in ("all", "sign", "exponent", "fraction", "value") if any(r["format"] == f and r["scope_kind"] == "weights" and r["fault_group"] == g for r in G)] for f in FORMATS}
    cols_w = [(f, g) for f in FORMATS for g in groups_w[f]]
    cols_s = [(f, g) for f in ("int8_pt", "int4_pt") for g in ("all", "sign", "exponent", "fraction")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.6), gridspec_kw={"width_ratios": [2.6, 1.25]}, sharey="row")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.2, wspace=0.05, hspace=0.7)

    def get(f, scope, g, k):
        return next(r for r in G if r["format"] == f and r["scope_kind"] == scope and r["fault_group"] == g and int(r["k"]) == k)
    for row, k in enumerate((1, 8)):
        for col, (scope, cols) in enumerate((("weights", cols_w), ("scales", cols_s))):
            ax = axes[row, col]
            xs = np.arange(len(cols)) + np.cumsum([0] + [0.5 if i and cols[i][0] != cols[i - 1][0] else 0 for i in range(1, len(cols))])
            for x, (f, g) in zip(xs, cols):
                r = get(f, scope, g, k)
                m, p90, wst = fl(r["drop_pp_mean"]), fl(r["drop_pp_p90"]), fl(r["drop_pp_worst"])
                ax.bar(x - 0.19, m, width=0.36, color=COL[f])
                ax.bar(x + 0.19, p90, width=0.36, color=COL[f], alpha=0.45)
                ax.plot([x], [wst], marker="v", ms=5, color=INK, ls="")
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylim(-0.3, 150)
            ax.set_yticks([0, 1, 10, 100])
            ax.set_yticklabels(["0", "1", "10", "100"])
            ab = {"exponent": "exp", "fraction": "frac", "value": "val"}
            nfs = [int(get(f, scope, g, k)["failures"]) for f, g in cols]
            ax.set_xticks(xs)
            ax.set_xticklabels([ab.get(g, g) + (f"\n{n} nf" if n else "") for (_, g), n in zip(cols, nfs)], rotation=0, fontsize=8.5)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{'stored weights' if scope == 'weights' else 'stored scale words (INT only)'}, k = {k}", pad=6, fontsize=11)
            if col == 0:
                ax.set_ylabel("accuracy drop (pp, symlog)")
            if row == 1:
                for f in dict.fromkeys(f for f, _ in cols):
                    pos = [x for x, (ff, _) in zip(xs, cols) if ff == f]
                    ax.text((pos[0] + pos[-1]) / 2, -0.27, LABEL[f], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9.5, color=COL[f], fontweight="bold")
    ax_leg = fig.add_axes([0.5, 0.01, 0.1, 0.02])
    ax_leg.axis("off")
    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=MUTED, label="mean"), Line2D([], [], marker="s", ls="", color=MUTED, alpha=0.45, label="p90"),
                        Line2D([], [], marker="v", ls="", color=INK, label="worst trial")], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    title(fig, "Fig. 4 - Fault robustness per format, fault group and k (global screening, 20 trials, first 1,000 validation images)",
          "k random bit flips anywhere in the 11.2 M stored weights (left) or in the 21 stored float32 scales (right); bar = mean drop, pale bar = p90, triangle = worst trial;\n"
          "exp = exponent, frac = fraction, val = value bits; nf = trials with non-finite output. Weight flips are harmless except FP/E4M11 exponent bits (FP16 worst); flips in the scale words of the INT formats\n"
          "destroy the model (sign, exponent) and dominate the integer formats' risk.")
    return save(fig, figdir, "fig4_fault_robustness")


def fig5(D, figdir):
    lfh = D["lfh"]
    cols = []
    for f in FORMATS:
        for g in ("all", "sign", "exponent", "fraction", "value"):
            key = f"{f}|weights|{g}|k8|mean_drop_pp"
            if key in lfh[0]:
                cols.append((f, g, key))
    scol = [(f, "all", f"{f}|scales|all|k1|mean_drop_pp") for f in ("int8_pt", "int4_pt")]
    M = np.array([[fl(r[k]) for _, _, k in cols] for r in lfh])
    S = np.array([[fl(r[k]) for _, _, k in scol] for r in lfh])
    fig = plt.figure(figsize=(16, 8.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[len(cols), len(scol) + 0.4, 0.45], left=0.135, right=0.955, top=0.8, bottom=0.12, wspace=0.12)
    a1, a2, ac = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    norm = PowerNorm(0.5, vmin=0, vmax=60)
    for ax, A, cs in ((a1, M, cols), (a2, S, scol)):
        im = ax.imshow(np.clip(A, 0, None), aspect="auto", cmap=SEQ, norm=norm)
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([g for _, g, _ in cs], rotation=0, fontsize=8.5)
        ax.grid(False)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] >= 1:
                    ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="white" if A[i, j] > 12 else INK)
    a1.set_yticks(range(len(lfh)))
    a1.set_yticklabels([r["layer_name"] for r in lfh], fontsize=9)
    a2.set_yticks([])
    prev, s = None, 0
    for j, (f, g, _) in enumerate(cols + [(None, None, None)]):
        if f != prev:
            if prev is not None:
                a1.text((s + j - 1) / 2, len(lfh) + 0.35, LABEL[prev], ha="center", va="top", fontsize=10, fontweight="bold", color=COL[prev])
                a1.axvline(j - 0.5, color=SURF, lw=3)
            prev, s = f, j
    for j, (f, g, _) in enumerate(scol):
        a2.text(j, len(lfh) + 0.35, LABEL[f], ha="center", va="top", fontsize=10, fontweight="bold", color=COL[f])
    a1.set_title("Weights: 8 random flips (k = 8) in ONE layer; bit group", pad=8, fontsize=11)
    a2.set_title("Scale word: 1 flip (k = 1)", pad=8, fontsize=11)
    cb = fig.colorbar(im, cax=ac)
    cb.set_label("mean accuracy drop (pp), 20 trials, 1,000 images")
    title(fig, "Fig. 5 - Layer sensitivity: layer x format / bit group",
          "Each cell is the mean accuracy drop (pp) when the flips are confined to that layer and bit group (numbers shown where >= 1 pp; colour scale is square-root).\n"
          "Exponent bits are the dominant weight risk: FP16 up to 60 pp in most layers, E4M11-SN and FP8 mainly in conv1, layer1 and the downsample convolutions;\n"
          "sign and fraction bits and INT weight flips stay below 1 pp (exception: INT4 sign bit in conv1, 2 pp). A flipped INT scale word costs 1-25 pp in every layer (right).")
    return save(fig, figdir, "fig5_layer_sensitivity_heatmap")


def fig6(D, figdir):
    rows = table_tradeoff(D)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.4), sharex=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.76, bottom=0.2, wspace=0.26)
    prot_col, plain_col = "#eb6834", "#2a78d6"

    def style(r):
        return dict(color=prot_col if (r["protected_layers"] not in ("0", 0)) else plain_col, marker="s" if r["assignment_id"].startswith("U_") else "o")
    a = axes[0]
    for r in rows:
        a.scatter(r["avg_bits"], r["clean_drop_pp"], s=22, color=MUTED, alpha=0.45, zorder=2)
    front = sorted([r for r in rows if r["pareto_bits_vs_clean_drop"]], key=lambda r: r["avg_bits"])
    a.step([r["avg_bits"] for r in front], [r["clean_drop_pp"] for r in front], where="post", color=INK2, lw=1.2, zorder=3)
    a.scatter([r["avg_bits"] for r in front], [r["clean_drop_pp"] for r in front], s=40, color=INK, zorder=4)
    for r in front:
        a.annotate(r["assignment_id"], (r["avg_bits"], r["clean_drop_pp"]), textcoords="offset points", xytext=(4, 5), fontsize=7.5, color=INK2)
    a.set_ylabel("clean accuracy drop vs FP32 (pp)\n2,952 validation images")
    a.set_title(f"(a) all {len(rows)} assembled assignments", pad=6, fontsize=11)
    meas = [r for r in rows if r["fault_measured"]]
    for ax, key, lab, ttl in ((axes[1], "weights_k8_mean_drop_pp", "mean drop, 8 random weight flips (pp)", "(b) weight faults (k = 8)"),
                              (axes[2], "scales_k1_mean_drop_pp", "mean drop, 1 flipped scale word (pp)", "(c) scale-word fault (k = 1)")):
        for r in meas:
            ax.scatter(r["avg_bits"], max(r[key], 0), s=70 if r["pareto_measured_4d"] else 36, edgecolor=INK if r["pareto_measured_4d"] else "none", linewidth=1.1, zorder=3, **style(r))
        for r in meas:
            if r["frozen"] or (r["pareto_measured_4d"] and r[key] > (1.0 if key.startswith("scales") else 0.02)):
                ax.annotate(r["assignment_id"], (r["avg_bits"], max(r[key], 0)), textcoords="offset points", xytext=(5, 5), fontsize=7.5, color=INK2)
        fr = [r for r in meas if r["frozen"]]
        for r in fr:
            ax.scatter(r["avg_bits"], max(r[key], 0), s=260, marker="*", color=INK, zorder=5)
        ax.set_yscale("symlog", linthresh=0.05 if key.startswith("weights") else 1)
        ax.set_ylabel(lab)
        ax.set_title(ttl, pad=6, fontsize=11)
    for ax in axes:
        ax.set_xlabel("stored bits per weight (payload + scales + protection)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=plain_col, label="no protection"), Line2D([], [], marker="o", ls="", color=prot_col, label="scale_tmr on its INT layers"),
                        Line2D([], [], marker="s", ls="", color=MUTED, label="uniform reference (square)"), Line2D([], [], marker="o", ls="", mec=INK, mfc="none", ms=9, label="Pareto candidate (outlined)"),
                        Line2D([], [], marker="*", ls="", color=INK, ms=14, label=f"frozen {D['assignment']['assignment_id']}")], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, fontsize=9.5)
    title(fig, "Fig. 6 - Trade-off: storage versus clean accuracy and residual fault risk",
          "(a) all assembled assignments, black = Pareto front of (bits, clean drop). (b, c) the 31 assignments with fault trials (100 trials per distribution, 1,000 images): outlined points are not dominated in\n"
          "(bits, clean drop, weight-fault drop, scale-word drop). Protection removes the scale-word failure at 8 bytes per layer; weight-fault means below 0.05 pp are within trial noise.")
    return save(fig, figdir, "fig6_tradeoff_pareto")


def fig7(D, figdir):
    LM = layer_metrics(D)
    A = table_alignment(LM)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.75, bottom=0.17, wspace=0.06)
    n = len(FORMATS)
    for ax, kk in zip(axes, ("8", "1")):
        for i, f in enumerate(FORMATS):
            vals = []
            for key, _ in PREDICTORS:
                r = next(x for x in A if x["format"] == f and x["k"] == kk and x["predictor"] == key)
                vals.append(r["spearman_rho"])
            xs = np.arange(len(PREDICTORS)) + (i - (n - 1) / 2) * 0.15
            ax.bar(xs, vals, width=0.14, color=COL[f], label=LABEL[f])
        crit = stats.t.ppf(0.975, 19) / math.sqrt(19 + stats.t.ppf(0.975, 19) ** 2)
        ax.axhspan(-crit, crit, color=GRID, alpha=0.8, zorder=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(PREDICTORS)))
        ax.set_xticklabels(["layer size", "weight std", "max |w|", "SQNR of format", "depth"], fontsize=9.5)
        ax.set_ylim(-1.02, 1.02)
        ax.set_title(f"risk of k = {kk} flips confined to ONE layer", pad=6, fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Spearman rank correlation across the 21 layers")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[f]) for f in FORMATS], labels=[LABEL[f] for f in FORMATS], loc="lower center", ncol=5, frameon=False, fontsize=9.5)

    title(fig, "Fig. 7 - Does layer risk align with layer size, weight distribution or quantization error?",
          "Risk = measured mean accuracy drop when the flips are confined to one layer (any bit); grey band = |rho| < 0.43, not significant at 5 % (n = 21).\n"
          "With k = 8, risk falls with layer size and depth and rises with weight std (conv1 and layer1 are the most sensitive). SQNR associates positively (layers with higher SQNR carry MORE risk)\n"
          "for E4M11-SN, FP8 and INT4 and not at all for FP16 and INT8, so more quantization error does not mean more risk. Size, std and depth are mutually correlated: associations, not separate causes.")
    return save(fig, figdir, "fig7_risk_alignment"), A, LM


def fig8(D, LM, figdir):
    fig, axes = plt.subplots(1, 5, figsize=(18, 5.6), sharex=True)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.7, bottom=0.15, wspace=0.28)
    for ax, f in zip(axes, FORMATS):
        sub = [r for r in LM if r["format"] == f]
        x = [r["num_weights"] for r in sub]
        y = [max(r["risk_k8_mean_drop_pp"], 0) for r in sub]
        yraw = [r["risk_k8_mean_drop_pp"] for r in sub]
        ax.scatter(x, y, s=34, color=COL[f])
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylim(bottom=-0.03)
        for r in sorted(sub, key=lambda r: -r["risk_k8_mean_drop_pp"])[:3]:
            ax.annotate(short(r["layer_name"]), (r["num_weights"], max(r["risk_k8_mean_drop_pp"], 0)), textcoords="offset points", xytext=(4, 4), fontsize=8, color=INK2)
        rho = stats.spearmanr(x, yraw)[0] if len(set(yraw)) > 1 else float("nan")
        ax.set_title(f"{LABEL[f]}  (rho = {rho:+.2f})", pad=6, fontsize=11)
        ax.set_xlabel("weights in the layer")
    axes[0].set_ylabel("mean drop, 8 flips in the layer (pp)")
    title(fig, "Fig. 8 - Layer risk versus layer size (k = 8 flips confined to the layer)",
          "One point per layer (21). The three highest-risk layers are named. The negative slope is partly arithmetic: 8 flips in a layer of 1,728 weights hit a far larger share of it than 8 flips in 2.4 M weights;\n"
          "the end-to-end metric (flips spread over the whole model) weights each layer by its size instead (R_exposure, Stage 5).")
    return save(fig, figdir, "fig8_risk_vs_layer_size")


def fig9(D, figdir):
    P = table_protection(D)
    fig, ax = plt.subplots(figsize=(15, 6.2))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.25)
    xs = np.arange(len(P))
    ax.bar(xs - 0.2, [r["scales_k1_mean_drop_pp_unprotected"] for r in P], width=0.38, color="#2a78d6", label="datatype only (no protection)")
    ax.bar(xs + 0.2, [r["scales_k1_mean_drop_pp_protected"] for r in P], width=0.38, color="#eb6834", label="datatype + scale_tmr")
    ax.scatter(xs - 0.2, [r["scales_k1_worst_drop_pp_unprotected"] for r in P], marker="v", color=INK, s=28, label="worst trial (unprotected)", zorder=4)
    ax.scatter(xs + 0.2, [r["scales_k1_worst_drop_pp_protected"] for r in P], marker="v", color=INK, s=28, zorder=4)
    for x, r in zip(xs, P):
        ax.text(x - 0.2, r["scales_k1_mean_drop_pp_unprotected"] + 1.2, f"{r['scales_k1_mean_drop_pp_unprotected']:.1f}", ha="center", fontsize=8)
        ax.text(x + 0.2, 4.0, f"{r['scales_k1_mean_drop_pp_protected']:.1f}", ha="center", fontsize=8.5, color="#b0451a", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['unprotected_id']}\nvs {r['protected_id']}\n{r['avg_bits_protected']:.2f} bit\n+{r['protection_overhead_bytes']} B" for r in P], fontsize=8.5)
    ax.set_ylabel("mean accuracy drop for ONE flipped scale bit (pp)")
    ax.set_ylim(0, 110)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=10)
    ax.grid(axis="x", visible=False)
    title(fig, "Fig. 9 - Datatype-only versus datatype + protection (same layer formats, measured end to end)",
          f"{len(P)} pairs of assembled models that differ only by scale_tmr on their INT layers (labels: ids, stored bits, extra bytes). Clean accuracy is identical in every pair; the 1-bit scale fault\n"
          "costs 9-25 pp on average (worst trial 88-96 pp) without protection and 0.0 pp with it. The fault is injected into the stored scale bytes (100 trials per model, 1,000 images).")
    return save(fig, figdir, "fig9_protection_pairs"), P


def fig10(D, figdir):
    V = [r for r in table_val_test(D) if not r["item"].startswith("FP32")]      # FP32 against itself is 0 by definition (kept in the table)
    fig, ax = plt.subplots(figsize=(13, 6.3))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.78, bottom=0.12)
    ys = np.arange(len(V))[::-1]
    cap = 2.2
    for y, r in zip(ys, V):
        a, b = min(max(r["val_drop_pp"], -0.4), cap), min(max(r["test_drop_pp"], -0.4), cap)
        ax.plot([a, b], [y, y], color=GRID, lw=2.5, zorder=1)
        ax.scatter(a, y, s=70, color="#2a78d6", zorder=3)
        ax.scatter(b, y, s=70, marker="s", color="#eb6834", zorder=3)
        ax.text(max(a, b) + 0.06, y, f"val {r['val_drop_pp']:+.2f} / test {r['test_drop_pp']:+.2f}" + (" (clipped)" if abs(r["val_drop_pp"]) > 50 else ""), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['item']}\n{r['val_images']:,} val / {r['test_images']:,} test images" for r in V], fontsize=8.5)
    ax.set_xlim(-0.5, cap + 1.8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("accuracy drop vs FP32 on the same images (pp); negative = better than FP32")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#2a78d6", label="validation (used to choose)"), Line2D([], [], marker="s", ls="", color="#eb6834", label="official test (read once)")],
              loc="lower right", frameon=False)
    title(fig, "Fig. 10 - Validation-derived selection versus the final test result",
          "The assignment was frozen from validation measurements only; the test set was read once afterwards.\n"
          "Validation and test agree within 0.25 pp (a few images). Uniform formats are shown for reference:\n"
          "their test numbers were reported after Stage 2 was frozen and were not used for any decision.")
    return save(fig, figdir, "fig10_validation_vs_test")


# ------------------------------------------------------------------ summary
def md_table(rows, cols, heads, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for r in rows:
        out.append("| " + " | ".join((fmt.get(c, lambda v: str(v))(r[c])) for c in cols) + " |")
    return "\n".join(out)


def summary(D, tag, T1, FT, A, P, V, TR, outs):
    B, Q, FP, C = D["baseline_manifest"], D["quant_manifest"], D["fault_protocol"], D["campaign"]
    ck, ds, pre = B["checkpoint"], B["dataset"], B["preprocessing"]
    asg, ft = D["assignment"], D["final_test"]
    env = B["environment_at_evaluation"]
    mv = asg["measured_validation_screen"]
    fin = asg["measured_validation_final_trials"]["fault_end_to_end"]
    cons = asg["constraints"]
    sel = asg["selected"]
    lines = []
    w = lines.append
    w(f"# Final summary - {tag}")
    w("")
    w(f"Generated by `scripts/stage6_final_package.py` from the saved records under `{D['base'].as_posix()}`; every table and figure below is rebuilt from those files. Hashes: `final_manifest.json`.")
    w("")
    w("## 1. Model, checkpoint, data")
    w(f"- **Architecture**: {B['model']['definition']} (`{B['model']['arch_in_config']}`), {B['model']['stem']}, {B['model']['output_dim']} classes; {B['runbook_reference']['n_parameters_total']:,} parameters, "
      f"{B['runbook_reference']['n_quantizable_layers']} quantizable Conv2d/Linear tensors with {B['runbook_reference']['n_quantizable_weights']:,} weights ({B['runbook_reference']['fp32_quantizable_bytes']:,} bytes in FP32).")
    w(f"- **Checkpoint**: `{ck['file_name']}`, SHA-256 `{ck['sha256']}`, {ck['size_bytes']:,} bytes, epoch {ck['epoch']}; trained from scratch in this project ({B['training_config']['epochs']} epochs, SGD lr {B['training_config']['lr']}, "
      f"weight decay {B['training_config']['weight_decay']}, batch {B['training_config']['batch_size']}, seed {B['training_config']['seed']}).")
    w(f"- **Data**: {ds['name']}; training pool {ds['train_pool']:,} images, validation holdout {ds['val_size']:,} (split seed {ds['split_seed']}, rule `{ds['split_rule']}`, validation-index SHA-256 `{ds['val_indices_sha256']}`), official test set {ds['test_images']:,} images "
      f"({ds['test_set']}).")
    w(f"- **Preprocessing**: normalization mean {pre['normalization_mean']}, std {pre['normalization_std']}, {pre['image_size']}, {pre['channel_order']}, no resize; FP32 evaluation, batch {B['evaluation']['batch_size']}.")
    w(f"- **Environment**: Python {env['python']}, torch {env['torch']}, CUDA {env['cuda_version']}, {env['gpu']}.")
    w(f"- **FP32 baseline**: validation {T1[0]['val_accuracy_pct']:.2f} % ({T1[0]['val_images']:,} images), test {T1[0]['test_accuracy_pct']:.2f} % ({T1[0]['test_images']:,} images).")
    w("")
    w("## 2. Candidate formats and quantization scope")
    w(f"- **Scope**: {Q['scope']}.")
    w(f"- **Independence**: {Q['independence']}. Excluded: " + "; ".join(f"{k}: {v}" for k, v in Q["formats_excluded"].items()) + ".")
    w("- **Exact variants**: " + "; ".join(f"{LABEL[r['format']]} = {r['variant']} ({r['role']})" for r in T1[1:]) + ".")
    w(f"- Frozen Stage 2 configuration SHA-256 `{Q['frozen_config_sha256']}`.")
    w("")
    w("## 3. Fault model")
    w(f"- **Model**: {FP['fault_model']}.")
    w("- **Fault groups and bit mapping** (bit 0 = least significant bit of the stored word):")
    for f, v in FP["formats"].items():
        gr = {g: b for g, b in v["groups"].items() if g != "all"}
        w(f"  - {LABEL.get(f, f)} ({v['bits_per_element']}-bit {v['kind']}, {v['layout'] if 'layout' in v else ''}): " + "; ".join(f"{g} = bits {b}" for g, b in gr.items()) + ".")
    w(f"- **k values**: {FP['fault_counts_k']}; {FP['sampling_policy']['locations']}.")
    w(f"- **Trials**: screening {C['settings']['trials_per_configuration']} per (layer, format, group, k) and per global configuration on the first {C['settings']['validation_images']:,} validation images; "
      f"end-to-end assignments 100 per distribution (1,000 images); frozen assignment 300 per distribution on validation and 100 per distribution on the test set ({ft['test_samples']:,} images).")
    w(f"- **Non-finite policy**: {FP['non_finite_policy']['accuracy_convention']}; trials are kept and counted as failures. **Restoration**: {FP['restoration_policy']}")
    w("- **Stage 5 distributions**: `weights_k8` = 8 distinct weights anywhere in the assembled model, bit uniform over each weight's own stored word (primary); `scales_k1` = 1 flipped bit uniform over all stored scale bits "
      "(32 per unprotected INT layer, 3 x 32 per `scale_tmr` layer) (second metric, never merged into the first).")
    w("")
    w("## 4. Evaluation sample counts")
    w(f"Validation 5,000 (uniform-format clean accuracy); validation images 0-2,047 for Q (exact Fisher), 2,048-4,999 ({D['e2e_metrics']['fp32_reference']['val_samples']:,}) for the clean accuracy of assembled assignments; "
      f"first 1,000 validation images for every fault trial; official test {ft['test_samples']:,} (read once for the frozen assignment, plus the uniform formats reported after Stage 2 was frozen).")
    w("")
    w("## 5. Required comparisons (runbook 8.1)")
    w("### 5.1 FP32 versus each candidate format (table `final_tables/*_format_comparison.csv`, Fig. 1-3)")
    w(md_table(T1, ["variant", "val_accuracy_pct", "val_drop_pp_vs_fp32", "test_accuracy_pct", "test_drop_pp_vs_fp32", "val_delta_ce_vs_fp32", "stored_bytes_weights", "bits_per_weight_actual", "compression_ratio_weights"],
               ["format", "val acc %", "val drop pp", "test acc %", "test drop pp", "val loss change", "stored bytes", "bit/weight", "compression x"],
               {"val_accuracy_pct": lambda v: f"{v:.2f}", "val_drop_pp_vs_fp32": lambda v: f"{v:+.2f}", "test_accuracy_pct": lambda v: f"{v:.2f}", "test_drop_pp_vs_fp32": lambda v: f"{v:+.2f}",
                "val_delta_ce_vs_fp32": lambda v: f"{v:+.4f}", "stored_bytes_weights": lambda v: f"{v:,}", "bits_per_weight_actual": lambda v: f"{v:.3f}", "compression_ratio_weights": lambda v: f"{v:.2f}"}))
    w("")
    w("### 5.2 Per-format fault robustness (global screening, k = 8 flips, any bit; full table incl. k = 1, groups, loss increase, non-finite rate: `final_tables/*_fault_robustness.csv`, Fig. 4)")
    rows = [r for r in FT if r["fault_group"] == "all" and r["k"] == "8"]
    w(md_table(rows, ["format", "scope_kind", "drop_pp_mean", "drop_pp_p90", "drop_pp_worst", "loss_increase_mean", "loss_increase_p90", "failures", "n_trials"],
               ["format", "target", "mean drop pp", "p90 pp", "worst pp", "mean loss increase", "p90 loss increase", "non-finite failures", "trials"],
               {"drop_pp_mean": lambda v: f"{fl(v):.3f}", "drop_pp_p90": lambda v: f"{fl(v):.3f}", "drop_pp_worst": lambda v: f"{fl(v):.2f}", "loss_increase_mean": lambda v: f"{fl(v):.4f}", "loss_increase_p90": lambda v: f"{fl(v):.4f}"}))
    w("")
    w("### 5.3 Layer-wise sensitivity (Fig. 5, 7, 8; tables `*_layer_sensitivity.csv`, `*_risk_alignment.csv`)")
    sig = [r for r in A if r["significant_5pct"] and r["k"] == "8"]
    w("Significant (5 %, n = 21 layers) Spearman associations of the k = 8 layer risk:")
    for f in FORMATS:
        mine = [r for r in sig if r["format"] == f]
        w(f"- {LABEL[f]}: " + (", ".join(f"{r['predictor_label']} {r['spearman_rho']:+.2f}" for r in mine) if mine else "none"))
    sig1 = [r for r in A if r["significant_5pct"] and r["k"] == "1"]
    w("For k = 1 the significant associations are: " + (", ".join(f"{LABEL[r['format']]} / {r['predictor_label']} {r['spearman_rho']:+.2f}" for r in sig1) if sig1 else "none") + ".")
    w("High-risk layers (top 3 by k = 8 mean drop): " + "; ".join(f"{LABEL[f]}: " + ", ".join(r["layer_name"] for r in sorted([x for x in layer_metrics(D) if x['format'] == f and x['top3_risk_k8']], key=lambda x: -x['risk_k8_mean_drop_pp'])) for f in FORMATS) + ".")
    w("Reading rules: the metric is the mean accuracy drop when all k flips fall in ONE layer, so small layers receive a larger share of flips per weight (part of the size association is arithmetic); layer size, weight std and depth are "
      "mutually correlated in a ResNet, so the associations do not separate causes; SQNR is a weight-space error and is not a consistent predictor of risk.")
    w("")
    w("### 5.4 Datatype-only versus datatype + protection (Fig. 9, `*_protection_pairs.csv`)")
    w("Only the implemented and measured protection is compared: `scale_tmr` (float32 scale word of an INT layer stored three times, bit-wise majority on read, +8 bytes per protected layer). "
      "No other protection (parity, ECC / SECDED on payload, protection of integer codes or of FP exponent bits) is implemented or claimed.")
    w(md_table(P, ["unprotected_id", "protected_id", "avg_bits_protected", "protection_overhead_bytes", "clean_drop_pp_unprotected", "clean_drop_pp_protected", "scales_k1_mean_drop_pp_unprotected", "scales_k1_mean_drop_pp_protected", "weights_k8_mean_drop_pp_unprotected", "weights_k8_mean_drop_pp_protected"],
               ["datatype-only", "+ scale_tmr", "bit/weight", "extra bytes", "clean drop pp (plain)", "clean drop pp (prot.)", "scale-word drop pp (plain)", "scale-word drop pp (prot.)", "weight-fault drop pp (plain)", "weight-fault drop pp (prot.)"],
               {"avg_bits_protected": lambda v: f"{v:.2f}", "clean_drop_pp_unprotected": lambda v: f"{v:.3f}", "clean_drop_pp_protected": lambda v: f"{v:.3f}", "scales_k1_mean_drop_pp_unprotected": lambda v: f"{v:.2f}",
                "scales_k1_mean_drop_pp_protected": lambda v: f"{v:.2f}", "weights_k8_mean_drop_pp_unprotected": lambda v: f"{v:.3f}", "weights_k8_mean_drop_pp_protected": lambda v: f"{v:.3f}"}))
    w("Within a pair the clean accuracy is identical (same layer formats). Differences in the weight-fault columns are sampling noise of 100 random draws, not an effect of the protection (it does not touch the integer codes): "
      "one rare high-magnitude trial moves a 100-trial mean by up to about 1 pp.")
    w("")
    w("### 5.5 Validation-derived selection versus final test (Fig. 10, `*_validation_vs_test.csv`)")
    w(f"Frozen assignment **{asg['assignment_id']}** (modes found: {asg['modes_found']}). Constraints: B = {cons['B_avg_bits_per_weight']} bits/weight, delta_clean = {cons['delta_clean_pp']} pp, tau_fault (weights_k8 mean drop) = "
      f"{cons['tau_fault_weights_k8_mean_drop_pp']} pp, tau_scale (scales_k1 mean drop) = {cons['tau_scale_scales_k1_mean_drop_pp']} pp; objective sum x (Q_norm + lambda R_norm), Q = `{asg['objective']['q_column']}`, R = `{asg['objective']['r_column']}`, "
      f"lambda = {asg['objective']['lambda']}, {asg['objective']['normalization']} normalization; solver: {asg['solver']}.")
    lay = {}
    for s in sel:
        key = s["datatype"] + ("+" + s["protection"] if s["protection"] != "none" else "")
        lay.setdefault(key, []).append(s["layer_name"])
    w("Layers: " + "; ".join(f"{k} x {len(v)}" + (f" ({', '.join(v)})" if len(v) <= 3 else "") for k, v in lay.items()) + f". Stored {mv['storage']['actual_stored_bytes']:,} bytes = {mv['storage']['avg_bits_per_weight']:.3f} bit/weight "
      f"(protection overhead {mv['storage']['protection_overhead_bytes']} bytes).")
    k8, s1 = fin["weights_k8"], fin["scales_k1"]
    tk8, ts1 = ft["test_fault"]["weights_k8"], ft["test_fault"]["scales_k1"]
    w(md_table([
        {"m": "clean accuracy", "val": f"{mv['clean_validation']['accuracy']:.2f} % ({mv['clean_validation']['samples']:,} images), drop {mv['clean_validation']['drop_pp_vs_fp32']:+.3f} pp",
         "test": f"{ft['assembled_test_accuracy']:.2f} % ({ft['test_samples']:,} images), drop {ft['test_clean_drop_pp_vs_fp32']:+.3f} pp (FP32 {ft['fp32_test_accuracy']:.2f} %)"},
        {"m": "weights_k8 mean / p90 / worst drop (pp)", "val": f"{k8['drop_pp_mean']:.3f} / {k8['drop_pp_p90']:.2f} / {k8['drop_pp_worst']:.2f} ({k8['n_trials']} trials, {k8['sample_count']:,} images)",
         "test": f"{tk8['drop_pp_mean']:.3f} / {tk8['drop_pp_p90']:.2f} / {tk8['drop_pp_worst']:.2f} ({tk8['n_trials']} trials, {tk8['sample_count']:,} images)"},
        {"m": "scales_k1 mean / worst drop (pp)", "val": f"{s1['drop_pp_mean']:.3f} / {s1['drop_pp_worst']:.2f} ({s1['n_trials']} trials)", "test": f"{ts1['drop_pp_mean']:.3f} / {ts1['drop_pp_worst']:.2f} ({ts1['n_trials']} trials)"},
        {"m": "non-finite failures (weights_k8 / scales_k1)", "val": f"{k8['failures']} / {s1['failures']}", "test": f"{tk8['failures']} / {ts1['failures']}"}],
        ["m", "val", "test"], ["measurement", "validation (decisions made here)", "official test (read once)"]))
    w("The scale-word result of the frozen assignment is a guarantee check, not sampling evidence: `scale_tmr` corrects every single flipped scale bit (exhaustive check in `optimization/tmr_exhaustive_check.json`).")
    w("")
    w("## 6. Figures (`figures/`)")
    for p in outs:
        w(f"- `{p.name}`")
    w("Earlier-stage figures (baseline distributions, Stage 4 screening, Stage 5 inputs) are in `plots_preview/<model-tag>/stage4/` and `stage5/`; the Stage 5 figures there were drawn from the datatype-only round-1 data.")
    w("")
    w("## 7. Exclusions, deviations and limitations")
    w("- **Excluded**: INT3 (optional, deferred); E4M11-FTZ is a comparison variant only and not a candidate; activations, biases and BatchNorm are not quantized or faulted.")
    w("- **Statistical depth**: Stage 4 is a screening (20 trials per cell instead of the paper-grade budget, first 1,000 validation images); layer-wise cells have a large standard error for bimodal faults; "
      "end-to-end distributions use 100 trials (300 for the frozen assignment), so weight-fault means below about 0.05 pp cannot be ranked and rare tail events (one trial of 85 pp in one assignment) are under-resolved.")
    w("- **Layer-additive estimates** (Q, R_exposure) are approximations; the final decision uses measured end-to-end values only. The weight-fault prediction ranks assignments weakly (Spearman about +0.35); Q ranks clean accuracy loss well (about +0.91) but not the cross-entropy change.")
    w("- **Protection**: only `scale_tmr` exists. Integer codes, E4M11 / FP exponent bits and any payload bits are unprotected; two flips at the same bit position in two copies of a scale word are not corrected (documented limit).")
    w("- **Deviations from the runbook**: (i) the J-FAMP objective, the fault distribution, tau_fault, delta_clean, B and tau_scale were fixed in `phase5/DECISIONS.md`; tau_scale is an addition to the runbook's constraint list; "
      "(ii) scale-word faults are an extension of the weight-only fault model and are reported as a second metric, never merged; (iii) clean accuracy of assembled assignments uses validation images 2,048-4,999 because Q uses images 0-2,047; "
      "(iv) the uniform-format test accuracies were reported once after Stage 2 was frozen and were not used for selection.")
    w("- **Scope**: one checkpoint, one seed, one dataset; weights only; the claims hold for this model and fault model.")
    w("")
    w("## 8. Rebuild")
    w("1. Stages 1-5 write the raw records (`baseline/`, `layer_profile/`, `quantization/`, `fault_injection/`, `fault_screening/`, `optimization/`).")
    w(f"2. `python scripts/stage6_final_package.py --base {D['base'].as_posix()} --model-tag {tag}` regenerates `final_tables/`, `figures/`, this file and `final_manifest.json`; compare the SHA-256 values with the committed manifest.")
    return "\n".join(lines) + "\n"


def main():
    global TAG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="reports/scratch/jfamp")
    ap.add_argument("--model-tag", default="resnet18_cifar10")
    args = ap.parse_args()
    TAG = args.model_tag
    base = Path(args.base)
    D = load(base)
    figdir, tabdir = base / "figures", base / "final_tables"
    figdir.mkdir(parents=True, exist_ok=True)

    T1 = table_formats(D)
    FT = table_fault(D)
    LM = layer_metrics(D)
    A = table_alignment(LM)
    P = table_protection(D)
    V = table_val_test(D)
    TR = table_tradeoff(D)
    tabs = {"format_comparison": T1, "fault_robustness": FT, "layer_sensitivity": LM, "risk_alignment": A, "protection_pairs": P, "validation_vs_test": V, "tradeoff_assignments": TR}
    tab_paths = []
    for name, rows in tabs.items():
        p = tabdir / f"{TAG}_{name}.csv"
        wcsv(p, rows)
        tab_paths.append(p)

    outs = [fig1(D, figdir), fig2(D, figdir), fig3(D, figdir), fig4(D, figdir), fig5(D, figdir), fig6(D, figdir)]
    p7, _, _ = fig7(D, figdir)
    outs += [p7, fig8(D, LM, figdir)]
    p9, _ = fig9(D, figdir)
    outs += [p9, fig10(D, figdir)]

    md = summary(D, TAG, T1, FT, A, P, V, TR, outs)
    sp = base / "final_summary.md"
    sp.write_text(md, encoding="utf8")
    print("wrote", sp)

    inputs = sorted(p for sub in ("baseline", "layer_profile", "quantization", "fault_injection", "fault_screening", "optimization") for p in (base / sub).glob("*") if p.is_file())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    manifest = {"model_tag": TAG, "base": base.as_posix(), "git_commit_of_this_script": commit,
                "inputs_sha256": {p.relative_to(base).as_posix(): sha256(p) for p in inputs},
                "outputs_sha256": {p.relative_to(base).as_posix(): sha256(p) for p in tab_paths + [sp]},
                "note": "figure PNGs are listed in final_summary.md; byte-identical PNGs are not guaranteed across matplotlib versions, the tables and the summary are"}
    (base / "final_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    print("wrote", base / "final_manifest.json")


if __name__ == "__main__":
    main()
