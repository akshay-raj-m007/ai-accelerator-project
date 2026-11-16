#!/usr/bin/env python
"""Figures and LaTeX tables for presentation/presentation.tex, built only from committed result files (no model, no GPU).

  python scripts/make_presentation_assets.py

Writes presentation/figures/*.png and presentation/generated_numbers.tex (every number of the slides that is not typed by hand).
Per-trial numbers: end_to_end_assignment_metrics.json (validation images 2,048-4,999; fault trials on a 1,000-image subset, 100 trials, 8 random weight bits),
sqnr_summary.csv (scripts/sqnr_report.py), final_test_evaluation.json (official test set, read once), fisher: ResNet-18 fisher_exact_val_layers.json, MobileNetV2 fisher_q.csv.
"""
import csv
import json
from pathlib import Path

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_mobilenetv2_summary import COL, GRID, INK, INK2, RC5, short as short_mnv2
from plot_stage5_summary import short as short_resnet

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "presentation"
FIG = OUT / "figures"
MAC = {"resnet18": "rn", "mobilenetv2": "mn"}  # LaTeX macro names may not contain digits
NICE = {"e4m11_sn": "E4M11-SN", "fp16": "FP16", "bf16": "BF16", "fp8_e4m3fn": "FP8 E4M3", "fp8_e5m2": "FP8 E5M2", "int8_pt": "INT8", "int4_pt": "INT4"}
ALIAS = {"E4M11SN": "e4m11_sn", "E4M11": "e4m11_sn", "FP16": "fp16", "BF16": "bf16", "FP8E5": "fp8_e5m2", "FP8N": "fp8_e4m3fn", "FP8": "fp8_e4m3fn", "INT8": "int8_pt", "INT4": "int4_pt"}

MODELS = {
    "resnet18": dict(name="ResNet-18", opt=ROOT / "reports/scratch/jfamp/optimization", sq=ROOT / "reports/scratch/jfamp/sqnr", final="A128"),
    "mobilenetv2": dict(name="MobileNetV2", opt=ROOT / "reports/mobilenetv2_cifar10/stage_06_fisher", sq=ROOT / "reports/mobilenetv2_cifar10/sqnr", final="A000"),
}


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def uniform_fmt(aid):
    b = aid[2:].split("_")
    return ALIAS[b[0]], (b[1] if len(b) > 1 else "none")


def load(key):
    cfg = MODELS[key]
    m = json.load(open(cfg["opt"] / "end_to_end_assignment_metrics.json"))
    asg = json.load(open(cfg["opt"] / "assignment.json"))
    test = json.load(open(cfg["opt"] / "final_test_evaluation.json"))
    sq = {r["assignment_id"]: r for r in rd(cfg["sq"] / "sqnr_summary.csv")}
    A = {a["assignment_id"]: a for a in m["assignments"]}
    return cfg, m, asg, test, sq, A


def fault_drop(a, dist="weights_k8"):
    f = a["fault_end_to_end"].get(dist)
    return None if not f else (f["drop_pp_mean"], f["drop_pp_p90"], f["drop_pp_worst"])


def fisher_per_weight(key, layers):
    if key == "resnet18":
        fl = json.load(open(ROOT / "reports/scratch/jfamp/optimization_round1/fisher_exact_val_layers.json"))["layers"]
        return np.array([fl[l + ".weight"]["trace_per_param"] for l in layers])
    nw, tr = {}, {}
    for r in rd(MODELS[key]["opt"] / "fisher_q.csv"):
        if r["format_key"] == "int4_pt":
            tr[r["layer_name"]] = float(r["fisher_trace"])
            nw[r["layer_name"]] = int(r["num_weights"])
    return np.array([tr[l] / nw[l] for l in layers])


def fig_fisher_assignment(key, cfg, asg):
    sel = asg["selected"]
    layers = [s["layer_name"] for s in sel]
    sh = short_resnet if key == "resnet18" else short_mnv2
    fw = fisher_per_weight(key, layers)
    L = len(layers)
    x = np.arange(L)
    col = [COL[s["datatype"]] for s in sel]
    plt.rcParams.update(RC5)
    fig, ax = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.7, 1], "hspace": 0.1})
    ax[0].bar(x, fw, color=col, width=0.72)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Fisher per weight (log)")
    ax[1].bar(x, [s["num_weights"] / 1e6 for s in sel], color=col, width=0.72)
    ax[1].set_ylabel("weights (millions)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([sh(l) for l in layers], rotation=90, fontsize=7.5 if L > 30 else 9)
    for a in ax:
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.set_xlim(-0.7, L - 0.3)
    PN = {"scale_tmr": "scale TMR", "secded": "SECDED", "none": "unprotected"}
    used = list(dict.fromkeys((s["datatype"], s["protection"]) for s in sel))
    ax[0].legend(handles=[Patch(color=COL[f], label=f"{NICE[f]} + {PN.get(p, p)}") for f, p in used], loc="upper right", frameon=False, fontsize=9, ncol=len(used))
    fig.suptitle(f"{cfg['name']}: Fisher sensitivity of each layer (input side left) and the format + protection the optimiser gave it", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(FIG / f"{key}_fisher_vs_assignment.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_before_after(key, cfg, m, A):
    fp = m["fp32_reference"]["val_accuracy"]
    ids = [i for i in A if i.startswith("U_") and "TMR" not in i and "SECDED" not in i]
    rows = []
    for i in ids:
        f, _ = uniform_fmt(i)
        rows.append((NICE[f], f, fp - A[i]["clean_validation"]["accuracy"], fault_drop(A[i])[0]))
    fin = A[cfg["final"]]
    rows.append(("FINAL assignment", None, fp - fin["clean_validation"]["accuracy"], fault_drop(fin)[0]))
    plt.rcParams.update(RC5)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.35})
    y = np.arange(len(rows))
    cols = [COL[r[1]] if r[1] else INK for r in rows]
    ax[0].barh(y, [r[2] for r in rows], color=cols, height=0.62)
    ax[1].barh(y, [r[3] for r in rows], color=cols, height=0.62)
    for a, idx, ttl in ((ax[0], 2, "before flips: accuracy loss from quantization (pp)"), (ax[1], 3, "after 8 random weight-bit flips: mean accuracy loss (pp)")):
        a.set_title(ttl, fontsize=10)
        a.set_yticks(y)
        a.set_yticklabels([r[0] for r in rows] if a is ax[0] else [])
        a.invert_yaxis()
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.tick_params(axis="y", length=0)
        a.axvline(0, color=INK2, lw=0.8)
        span = max(abs(r[idx]) for r in rows)
        for yi, r in zip(y, rows):
            v = r[idx]
            a.text(v + (0.01 * span if v >= 0 else -0.01 * span), yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
        a.set_xlim(min(0, min(r[idx] for r in rows)) - 0.12 * span, max(r[idx] for r in rows) + 0.2 * span)
    fig.suptitle(f"{cfg['name']}: uniform formats (no protection) vs the final assignment, before and after 8 bit flips", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(FIG / f"{key}_datatype_before_after_k8.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_headline(data):
    plt.rcParams.update(RC5)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw={"wspace": 0.38})
    names = [data[k]["cfg"]["name"] for k in data]
    x = np.arange(len(names))
    w = 0.34
    BL, OR = COL["e4m11_sn"], COL["fp16"]
    acc0 = [data[k]["test"]["fp32_test_accuracy"] for k in data]
    acc1 = [data[k]["test"]["assembled_test_accuracy"] for k in data]
    ax[0].bar(x - w / 2, acc0, w, color=BL, label="initial FP32")
    ax[0].bar(x + w / 2, acc1, w, color=OR, label="after allocation")
    for i in range(len(x)):
        ax[0].text(x[i] - w / 2, acc0[i] + 0.1, f"{acc0[i]:.2f}", ha="center", fontsize=8.5)
        ax[0].text(x[i] + w / 2, acc1[i] + 0.1, f"{acc1[i]:.2f}", ha="center", fontsize=8.5)
    ax[0].set_ylim(88, 97.5)
    ax[0].set_title("Test accuracy (%)", fontsize=11)
    mb0 = [data[k]["bytes0"] / 1e6 for k in data]
    mb1 = [data[k]["bytes1"] / 1e6 for k in data]
    ax[1].bar(x - w / 2, mb0, w, color=BL)
    ax[1].bar(x + w / 2, mb1, w, color=OR)
    for i in range(len(x)):
        ax[1].text(x[i] - w / 2, mb0[i] * 1.02, f"{mb0[i]:.2f}", ha="center", fontsize=8.5)
        ax[1].text(x[i] + w / 2, mb1[i] * 1.02 + 0.3, f"{mb1[i]:.2f}", ha="center", fontsize=8.5)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1, 120)
    ax[1].set_title("Weight storage (MB, log axis)", fontsize=11)
    sq = [float(data[k]["sq"][data[k]["cfg"]["final"]]["sqnr_clean_db"]) for k in data]
    ax[2].bar(x + w / 2, sq, w, color=OR)
    for i in range(len(x)):
        ax[2].text(x[i] - w / 2, 3, "+inf\n(no error)", ha="center", fontsize=8.5, color=INK2)
        ax[2].bar(x[i] - w / 2, 0.001, w, color=BL)
        ax[2].text(x[i] + w / 2, sq[i] + 0.5, f"{sq[i]:.1f}", ha="center", fontsize=8.5)
    ax[2].set_ylim(0, 32)
    ax[2].set_title("SQNR of the stored weights (dB)", fontsize=11)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(names)
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.savefig(FIG / "headline_initial_vs_final.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def tex_tables(data):
    L = []
    for key, d in data.items():
        cfg, m, asg, test, sq, A = d["cfg"], d["m"], d["asg"], d["test"], d["sq"], d["A"]
        fp = m["fp32_reference"]["val_accuracy"]
        rows = []
        order = [i for i in A if i.startswith("U_") and "TMR" not in i and "SECDED" not in i] + [cfg["final"]]
        for i in order:
            a = A[i]
            f = fault_drop(a)
            name = "\\textbf{Final (" + i + ")}" if i == cfg["final"] else NICE[uniform_fmt(i)[0]]
            s = sq[i]
            rows.append(f"{name} & {a['storage']['avg_bits_per_weight']:.2f} & {a['clean_validation']['accuracy']:.2f} & {f[0]:+.2f} & {f[2]:.2f} & {float(s['sqnr_clean_db']):.1f} & {float(s['weights_k8__sqnr_median_db']):.1f} \\\\")
        L.append("\\newcommand{\\" + key + "FormatTable}{%\n" + "\n".join(rows) + "}")
        n_w = sum(s["num_weights"] for s in asg["selected"])
        fin = A[cfg["final"]]
        by0 = n_w * 4
        by1 = fin["storage"]["actual_stored_bytes"]
        d["bytes0"], d["bytes1"] = by0, by1
        L.append("\\newcommand{\\" + key + "Bytes}{" + f"{by0:,}".replace(",", "\\,") + " & " + f"{by1:,}".replace(",", "\\,") + " & " + f"{100 * (1 - by1 / by0):.1f}" + "}")
        L.append("\\newcommand{\\" + key + "TestAcc}{" + f"{test['fp32_test_accuracy']:.2f} & {test['assembled_test_accuracy']:.2f} & {test['assembled_test_accuracy'] - test['fp32_test_accuracy']:+.2f}" + "}")
        L.append("\\newcommand{\\" + key + "ValAcc}{" + f"{fp:.2f} & {fin['clean_validation']['accuracy']:.2f} & {fin['clean_validation']['accuracy'] - fp:+.2f}" + "}")
        s = sq[cfg["final"]]
        L.append("\\newcommand{\\" + key + "Sqnr}{" + f"$+\\infty$ & {float(s['sqnr_clean_db']):.2f} & {float(s['weights_k8__sqnr_median_db']):.2f} & {float(s['weights_k8__sqnr_worst_finite_db']):.2f}" + "}")
        fl = fault_drop(fin)
        sc = fin["fault_end_to_end"]["scales_k1"]
        L.append("\\newcommand{\\" + key + "FaultFinal}{" + f"{fl[0]:+.3f} & {fl[1]:.2f} & {fl[2]:.2f} & {sc['drop_pp_mean']:.2f}" + "}")
        # layer table
        sel = asg["selected"]
        layers = [x["layer_name"] for x in sel]
        sh = short_resnet if key == "resnet18" else short_mnv2
        fw = fisher_per_weight(key, layers)
        rank = (-fw).argsort().argsort() + 1
        lines = []
        for x, v, r in zip(sel, fw, rank):
            p = {"none": "--", "scale_tmr": "scale TMR", "secded": "SECDED"}.get(x["protection"], x["protection"])
            lines.append(f"{sh(x['layer_name']).replace('_', ' ')} & {x['num_weights']:,} & {v:.1e} & {int(r)} & {NICE[x['datatype']]} & {p} \\\\".replace(",", "\\,", 0))
        L.append("\\newcommand{\\" + key + "LayerRows}{%\n" + "\n".join(lines) + "}")
        cnt = {}
        for x in sel:
            k = NICE[x["datatype"]] + (" + " + {"scale_tmr": "scale TMR", "secded": "SECDED"}.get(x["protection"], x["protection"]) if x["protection"] not in ("none", "") else "")
            cnt[k] = cnt.get(k, 0) + 1
        L.append("\\newcommand{\\" + key + "Mix}{" + "; ".join(f"{v} $\\times$ {k}" for k, v in cnt.items()) + "}")
    txt = "\n".join(L) + "\n"
    for k, v in (("resnet18", "rn"), ("mobilenetv2", "mn")):  # LaTeX macro names may not contain digits
        txt = txt.replace("\\newcommand{\\" + k, "\\newcommand{\\" + v)
    (OUT / "generated_numbers.tex").write_text(txt, encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = {}
    for k in MODELS:
        cfg, m, asg, test, sq, A = load(k)
        data[k] = dict(cfg=cfg, m=m, asg=asg, test=test, sq=sq, A=A)
    tex_tables(data)
    for k, d in data.items():
        fig_fisher_assignment(k, d["cfg"], d["asg"])
        fig_before_after(k, d["cfg"], d["m"], d["A"])
    fig_headline(data)
    print("written to", OUT)


if __name__ == "__main__":
    main()
