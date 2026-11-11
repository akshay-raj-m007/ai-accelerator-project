"""Weight and activation distributions, clean vs faulted, for two formats (Stage 4 follow-up; needs the GPU machine's encoded weights).

For each format the worst RECORDED trial of one shared configuration (same layer, fault group and k, taken from fault_trials.jsonl)
is replayed exactly (same encoded elements and bits), so the figure shows a trial that is part of the Stage 4 evidence.
Default: formats FP16 and E4M11-SN, fault group exponent, k = 8, layer-wise scope. The layer is chosen automatically as the one with the
largest sum of the two formats' mean accuracy drops in fault_summary.csv (or give --layer).

Columns per format: (1) weights of the faulted layer, clean vs faulted, with each flipped element marked old -> new;
(2) activations (output of the faulted layer) clean vs faulted; (3) logits clean vs faulted; (4) largest |activation| per layer through the network.
Also writes distribution_stats.json with the exact numbers, and checks that the replayed trial reproduces the recorded fault accuracy.

  python scripts/plot_fault_distributions.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import _common  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fr import data as fdata
from fr import stage2 as S2
from fr import stage3 as S3
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import resolve_device

LABEL = {"e4m11_sn": "E4M11-SN", "fp16": "FP16", "fp8_e4m3fn": "FP8 E4M3FN", "int8_pt": "INT8", "int4_pt": "INT4"}
CLEAN_C, FAULT_C = "tab:blue", "tab:red"


# ------------------------------------------------------------------------------------------------ collection
@torch.inference_mode()
def collect(model, X, target: int, batch_size: int = 250, max_samples: int = 2_000_000, seed: int = 0):
    """Per quantizable layer: exact largest finite |output| and non-finite count; for ``target``: a sample of its output values; plus logits."""
    model.eval()
    mods = quantizable_modules(model)
    maxabs = [0.0] * len(mods)
    nonfin = [0] * len(mods)
    samples, logits = [], []
    gen = torch.Generator(device=X.device).manual_seed(seed)
    nb = (X.shape[0] + batch_size - 1) // batch_size
    per = max(max_samples // nb, 1000)
    hooks = []

    def mk(i):
        def hook(mod, inp, out):
            t = out.detach().float().reshape(-1)
            fin = torch.isfinite(t)
            nonfin[i] += int((~fin).sum())
            if fin.any():
                maxabs[i] = max(maxabs[i], float(t[fin].abs().max()))
            if i == target:
                tf = t[fin]
                if tf.numel() > per:
                    tf = tf[torch.randint(0, tf.numel(), (per,), generator=gen, device=tf.device)]
                samples.append(tf.cpu())
        return hook

    for i, (_, m) in enumerate(mods):
        hooks.append(m.register_forward_hook(mk(i)))
    try:
        for s in range(0, X.shape[0], batch_size):
            logits.append(model(X[s:s + batch_size]).detach().float().cpu())
    finally:
        for h in hooks:
            h.remove()
    lg = torch.cat(logits).reshape(-1)
    return {"maxabs": maxabs, "nonfinite": nonfin, "act": torch.cat(samples).numpy(),
            "logits": lg[torch.isfinite(lg)].numpy(), "logits_nonfinite": int((~torch.isfinite(lg)).sum())}


# ------------------------------------------------------------------------------------------------ plotting helpers
def symbins(values_list, lin=0.05, n_lin=80, n_log=60):
    """Bin edges: linear inside [-lin, lin], logarithmic outside, covering every array in ``values_list``."""
    hi = max([float(np.max(np.abs(v))) for v in values_list if len(v)] + [lin * 2])
    pos = np.logspace(np.log10(lin), np.log10(hi * 1.001), n_log)
    return np.concatenate([-pos[::-1], np.linspace(-lin, lin, n_lin)[1:-1], pos]), lin


def hist_pair(ax, clean, fault, title, xlabel, lin, marks=None, note=""):
    edges, lin = symbins([clean, fault], lin)
    for v, c, lab in ((clean, CLEAN_C, "clean"), (fault, FAULT_C, "faulted")):
        counts, _ = np.histogram(v, bins=edges)
        ax.stairs(counts, edges, fill=(c == CLEAN_C), alpha=0.4 if c == CLEAN_C else 1.0, color=c, lw=1.2,
                  label=f"{lab} (n={len(v):,}, max|v|={np.max(np.abs(v)):.3g})" if len(v) else lab)
    ax.set_xscale("symlog", linthresh=lin)
    ax.set_yscale("log")
    if marks is not None:
        for old, new in marks:
            ax.plot([old, new], [1, 1], color="grey", lw=0.8, zorder=2)
        ax.scatter([o for o, _ in marks], [1] * len(marks), marker="o", facecolors="none", edgecolors="k", s=26, zorder=3, label="flipped element: old")
        ax.scatter([n for _, n in marks], [1] * len(marks), marker="x", color="darkred", s=34, zorder=4, label="flipped element: new")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel + f"  (symlog axis, linear inside +-{lin:g})", fontsize=8)
    ax.set_ylabel("count (log)", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")
    if note:
        ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=6.5, va="top")
    ax.grid(alpha=0.25)


# ------------------------------------------------------------------------------------------------ main
def pick_layer(rows, fmts, group, k):
    d = {}
    for r in rows:
        if r["scope_kind"] == "weights" and r["fault_group"] == group and int(r["k"]) == k and r["format"] in fmts and r["layer_name"] != "GLOBAL":
            d.setdefault(r["layer_name"], {})[r["format"]] = float(r["drop_pp_mean"])
    cand = {n: sum(v.values()) for n, v in d.items() if len(v) == len(fmts)}
    return max(cand, key=cand.get)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--formats", nargs=2, default=["fp16", "e4m11_sn"])
    ap.add_argument("--group", default="exponent")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--layer", default=None, help="module name, default: automatic (see module docstring)")
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--screening-dir", default=None, help="folder with fault_trials.jsonl / fault_summary.csv")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--lin", type=float, default=0.05, help="half-width of the linear part of the symlog axes")
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    sdir = Path(args.screening_dir) if args.screening_dir else base / "fault_screening"
    out = Path(args.out_dir) if args.out_dir else sdir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    man2 = json.load(open(qdir / "quantizer_manifest.json"))
    ck = Path(args.checkpoint)
    if man2["checkpoint"]["sha256"] != sha256_file(ck):
        sys.exit("checkpoint SHA-256differs from the one Stage 2 was frozen on")
    rows = list(csv.DictReader(open(sdir / "fault_summary.csv")))
    recs = [json.loads(l) for l in open(sdir / "fault_trials.jsonl")]
    model, ckpt = load_model(ck)
    model.eval()
    mods = quantizable_modules(model)
    names = [n for n, _ in mods]
    layer_names = [f"{n}.weight" for n in names]
    lname = args.layer or pick_layer(rows, args.formats, args.group, args.k)
    li = names.index(lname)
    print(f"layer {lname} (id {li}), group {args.group}, k = {args.k}, formats {args.formats}")

    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=args.n_images)
    X, y = load_tensors(loader, device)

    fig, axes = plt.subplots(2, 4, figsize=(23, 9.5))
    stats = {"layer": lname, "layer_id": li, "group": args.group, "k": args.k, "n_images": int(y.numel()), "formats": {}}
    for row, fmt in enumerate(args.formats):
        cand = [r for r in recs if r["format"] == fmt and r["scope_kind"] == "weights" and r["layer_name"] == lname
                and r["fault_group"] == args.group and r["requested_k"] == args.k]
        if len(cand) == 0:
            sys.exit(f"no recorded trials for {fmt}/{lname}/{args.group}/k{args.k}")
        rec = max(cand, key=lambda r: r["clean_accuracy"] - r["fault_accuracy"])
        stored = S2.load_encoded(qdir / man2["formats"][fmt]["encoded_file"], layer_names)
        if S2.content_sha256(stored) != man2["formats"][fmt]["content_sha256"]:
            sys.exit(f"{fmt}: encoded weights differ from the frozen content hash")
        codec = S2.make_codecs(True)[fmt]
        inj = S3.EncodedFaultInjector(model, S3.EncodedStore(fmt, stored), codec, device)
        clean_eval = S3.evaluate(inj.model, X, y)
        clean = collect(inj.model, X, li)
        w_clean = inj.flat[li].detach().cpu().numpy().astype(np.float64).copy()

        flips = [(f[1], f[2], f[3]) for f in rec["flips"] if f[0] == "w"]
        elems = sorted({e for l, e, b in flips})
        before = {e: float(inj.flat[li][e]) for e in elems}
        inj.inject(weight_flips=flips)
        try:
            after = {e: float(inj.flat[li][e]) for e in elems}
            w_fault = inj.flat[li].detach().cpu().numpy().astype(np.float64).copy()
            fault_eval = S3.evaluate(inj.model, X, y)
            fault = collect(inj.model, X, li)
        finally:
            inj.restore()
        replay_ok = None
        if int(y.numel()) == rec["sample_count"]:
            replay_ok = abs(fault_eval["accuracy"] - rec["fault_accuracy"]) < 1e-6
            print(f"{fmt}: replayed trial {rec['trial_id']} fault accuracy {fault_eval['accuracy']:.2f} vs recorded {rec['fault_accuracy']:.2f} -> "
                  f"{'REPRODUCED' if replay_ok else 'MISMATCH'}")
        marks = [(before[e], after[e]) for e in elems]
        tag = f"{LABEL[fmt]}: trial {rec['trial_id'].split('|')[-1]}  (accuracy {clean_eval['accuracy']:.1f} -> {fault_eval['accuracy']:.1f} %)"

        hist_pair(axes[row, 0], w_clean, w_fault, f"{tag}\nweights of {lname}, clean vs faulted", "weight value", args.lin, marks=marks,
                  note=f"{len(elems)} elements changed of {w_clean.size:,}")
        hist_pair(axes[row, 1], clean["act"], fault["act"], f"{LABEL[fmt]}: activations = output of {lname}", "activation value", args.lin,
                  note=f"non-finite outputs: clean {clean['nonfinite'][li]}, faulted {fault['nonfinite'][li]}")
        hist_pair(axes[row, 2], clean["logits"], fault["logits"], f"{LABEL[fmt]}: logits ({int(y.numel())} images x 10)", "logit value", args.lin,
                  note=f"non-finite logits: clean {clean['logits_nonfinite']}, faulted {fault['logits_nonfinite']}")
        ax = axes[row, 3]
        x = np.arange(len(names))
        ax.plot(x, np.maximum(clean["maxabs"], 1e-12), "o-", color=CLEAN_C, label="clean", ms=4)
        ax.plot(x, np.maximum(fault["maxabs"], 1e-12), "s-", color=FAULT_C, label="faulted", ms=4)
        bad = [i for i, n in enumerate(fault["nonfinite"]) if n]
        if bad:
            ax.scatter(bad, [max(fault["maxabs"][i], 1e-12) for i in bad], marker="*", s=150, color="k", zorder=5, label="layers with non-finite output")
        ax.axvline(li, color="grey", ls="--", lw=1, label=f"faulted layer ({lname})")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_ylabel("largest finite |output| of the layer", fontsize=8)
        ax.set_title(f"{LABEL[fmt]}: largest |activation| per layer through the network", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        stats["formats"][fmt] = {
            "trial_id": rec["trial_id"], "recorded_drop_pp": rec["clean_accuracy"] - rec["fault_accuracy"], "replayed_fault_accuracy": fault_eval["accuracy"],
            "clean_accuracy": clean_eval["accuracy"], "replay_reproduces_record": replay_ok,
            "flipped": [{"element": e, "bit(s)": [b for l, ee, b in flips if ee == e], "old": before[e], "new": after[e]} for e in elems],
            "weights_clean": {"max_abs": float(np.abs(w_clean).max()), "std": float(w_clean.std())},
            "weights_faulted": {"max_abs": float(np.abs(w_fault).max()), "std": float(w_fault.std())},
            "act_max_abs_clean": clean["maxabs"], "act_max_abs_faulted": fault["maxabs"], "act_nonfinite_faulted": fault["nonfinite"],
            "logits_nonfinite_faulted": fault["logits_nonfinite"]}
        del inj
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fig.suptitle(f"Worst recorded trial of {args.group} k={args.k} faults in {lname} (Stage 4, {int(y.numel())} validation images): "
                 f"{LABEL[args.formats[0]]} vs {LABEL[args.formats[1]]}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = out / f"fault_distributions_{args.formats[0]}_vs_{args.formats[1]}.png"
    fig.savefig(png, dpi=150)
    json.dump(stats, open(out / f"fault_distributions_{args.formats[0]}_vs_{args.formats[1]}.json", "w"), indent=1)
    print("written", png)


if __name__ == "__main__":
    main()
