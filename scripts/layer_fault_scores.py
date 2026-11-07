#!/usr/bin/env python
"""Plan B: per-layer scores S0 / S1 / S2 / S2n on the all-E4M11 model, and their rank correlations. See fr/layer_scores.py.

Held-out data only: Fisher = FIRST 2,048 validation images (the file sensitivity_heldout/fisher_true.pt), the measured
damage S1 is evaluated on the REMAINING 2,952 validation images (`--val-skip 2048`). The test set is not touched.

  python scripts/layer_fault_scores.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch \
      --fisher $FR_ROOT/results/scratch/sensitivity_heldout/fisher_true.pt

Writes results/<run>/layer_scores/{layer_scores.csv, layer_scores.json, layer_scores.md}.
Also computes the Fisher of the E4M11-QUANTIZED model (same 2,048 images) to check that it matches the FP32 Fisher.
"""
import argparse
import csv
import logging
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from fr import data as fdata, faults, mixed, sensitivity
from fr import faults_mixed as FM
from fr import layer_scores as LS
from fr.config import add_config_args, config_from_args
from fr.mixed_run import split_params
from fr.model import load_model
from fr.utils import dump_json, resolve_device


def fmt(x, d=4):
    return f"{x:.{d}g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fisher", required=True, help="held-out true-Fisher file (sensitivity_heldout/fisher_true.pt)")
    ap.add_argument("--out-dir", default=None, help="default results/<run>/layer_scores")
    ap.add_argument("--groups", nargs="+", default=["any", "exponent"])
    ap.add_argument("--k", type=int, default=16, help="bits struck per trial, in one layer only")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--sampling", choices=FM.SAMPLINGS, default="bits")
    ap.add_argument("--val-skip", type=int, default=2048)
    ap.add_argument("--skip-e4m11-fisher", action="store_true")
    ap.add_argument("--e4m11-mode", default="subnormal")
    ap.add_argument("--e4m11-bias", type=int, default=7)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    out = Path(args.out_dir) if args.out_dir else cfg.results_dir() / "layer_scores"
    out.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(args.checkpoint)
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    fisher = sensitivity.load_fisher(args.fisher)
    if fisher.get("source") not in (None, "val"):
        raise SystemExit(f"{args.fisher} is a {fisher.get('source')!r}-split Fisher; Plan B needs the held-out one")
    F = fisher["fisher"]

    qmodel, buffers = mixed.quantize_mixed(model, {}, args.e4m11_mode, args.e4m11_bias)
    inj = FM.MixedFaultInjector(qmodel, buffers, args.e4m11_mode, args.e4m11_bias, "clamp", device)
    assert inj.clean_weights_unchanged()
    names = inj.names
    counts = inj.counts
    res = {n: {"layer": n, "count": int(c)} for n, c in zip(names, counts)}

    # ---- S0
    for n in names:
        tr = float(F[n].double().sum())
        res[n].update(S0_trace=tr, S0_per_param=tr / res[n]["count"])
    # ---- S0 on the E4M11 model
    if not args.skip_e4m11_fisher:
        loader = fdata.get_val_loader(cfg.data_path(), norm, 64, val_size, split_seed, cfg.num_workers, False,
                                      subset=args.val_skip)
        r = sensitivity.compute_fisher(inj.model, loader, device, kind="true", seed=cfg.seed)
        for n in names:
            res[n]["S0_e4m11_trace"] = float(r["fisher"][n].double().sum())
        print(f"Fisher of the E4M11 model: {r['n_samples']} samples, mean NLL {r['mean_nll']:.4f}")
    # ---- S2, S2n
    for g in args.groups:
        for m in LS.predicted_flip_moments(inj, F, g):
            res[m["layer"]].update({f"S2_{g}": m["S2"], f"S2n_{g}": m["S2n"], f"max_delta_{g}": m["max_abs_delta"]})
    # ---- S1
    loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed,
                                  cfg.num_workers, False, subset=None, skip=args.val_skip)
    X, y = faults.load_tensors(loader, device)
    clean = faults.eval_tensors(inj.model, X, y)
    print(f"clean E4M11 on {X.shape[0]} held-out validation images: {clean['accuracy_percent']:.2f} %")
    for g in args.groups:
        for row in LS.measured_layer_damage(inj, X, y, g, args.k, args.trials, cfg.seed, args.sampling):
            res[row["layer"]].update({f"S1_{g}_{key}": v for key, v in row.items()
                                      if key not in ("layer", "count", "group", "k", "trials")})
    assert inj.clean_weights_unchanged()

    # ---- tables
    rows = [res[n] for n in names]
    keys = sorted({k for r in rows for k in r})
    with open(out / "layer_scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["layer", "count"] + [k for k in keys if k not in ("layer", "count")])
        w.writeheader()
        w.writerows(rows)

    corr = []
    for g in args.groups:
        for tgt in ("d_ce_mean", "kl_mean", "d_acc_mean"):
            s1 = [r[f"S1_{g}_{tgt}"] for r in rows]
            for label, key in (("S0 trace per parameter", "S0_per_param"), ("S0 trace (sum)", "S0_trace"),
                               ("S2 Fisher-predicted", f"S2_{g}"), ("S2n no Fisher", f"S2n_{g}")):
                corr.append({"group": g, "target": f"S1 {tgt}", "score": label,
                             "spearman": LS.spearman([r[key] for r in rows], s1)})
    dump_json({"clean": clean, "k": args.k, "trials": args.trials, "sampling": args.sampling,
               "n_eval": int(X.shape[0]), "layers": rows, "spearman": corr}, out / "layer_scores.json")

    L = ["# Per-layer scores on the all-E4M11 model (Plan B)", "",
         f"* S1 measured on {X.shape[0]} held-out validation images; clean accuracy {clean['accuracy_percent']:.2f} %; "
         f"k = {args.k} bits struck in one layer only, {args.trials} trials per layer, sampling `{args.sampling}`",
         "* Fisher: first 2,048 validation images (held out from S1). Definitions of S0, S1, S2, S2n: `src/fr/layer_scores.py`",
         ""]
    for g in args.groups:
        L += [f"## Group `{g}`", "",
              "| layer | weights | S0 tr(F)/N | S2 pred. | S2n (no F) | S1 dCE mean | S1 dCE median | S1 KL mean | S1 dAcc mean (pp) "
              "| S1 dAcc p90 | non-finite |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['layer']} | {r['count']} | {fmt(r['S0_per_param'])} | {fmt(r[f'S2_{g}'])} | {fmt(r[f'S2n_{g}'])} | "
                     f"{r[f'S1_{g}_d_ce_mean']:+.4f} | {r[f'S1_{g}_d_ce_median']:+.4f} | {fmt(r[f'S1_{g}_kl_mean'])} | "
                     f"{r[f'S1_{g}_d_acc_mean']:+.2f} | {r[f'S1_{g}_d_acc_p90']:+.2f} | {r[f'S1_{g}_non_finite']} |")
        L.append("")
    L += ["## Spearman rank correlation of each score with the measured damage S1 (21 layers)", "",
          "| group | S1 target | score | Spearman |", "|---|---|---|---|"]
    L += [f"| {c['group']} | {c['target']} | {c['score']} | {c['spearman']:+.3f} |" for c in corr]
    if "S0_e4m11_trace" in rows[0]:
        a = [r["S0_trace"] for r in rows]
        b = [r["S0_e4m11_trace"] for r in rows]
        ratio = np.array(b) / np.array(a)
        L += ["", "## Fisher of the FP32 model vs the E4M11-quantized model (layer traces)", "",
              f"* Spearman {LS.spearman(a, b):+.4f}; ratio E4M11/FP32 per layer: min {ratio.min():.3f}, "
              f"median {np.median(ratio):.3f}, max {ratio.max():.3f}"]
    (out / "layer_scores.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("written:", out)


if __name__ == "__main__":
    main()
