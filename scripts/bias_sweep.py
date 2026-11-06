#!/usr/bin/env python
"""Sweep the E4M11 exponent bias (and encoder mode) on a trained checkpoint.

For each (mode, bias): Conv+Linear SQNR (Eq. 6), max representable value, how many weights are
clipped / flushed, and (unless --no-accuracy) test accuracy. Exploratory: bias != 7 deviates from
the paper (Table 1). Choose the bias by a stated criterion, not by looking at the paper's SQNR.

  python scripts/bias_sweep.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch_bias
"""
import argparse
import logging

import numpy as np

import _common  # noqa: F401
from fr import data as fdata
from fr import e4m11
from fr.config import add_config_args, config_from_args
from fr.evaluate import evaluate
from fr.model import load_model
from fr.quantize import conv_linear_weights, quantize_model
from fr.sqnr import sqnr_report
from fr.utils import dump_json, resolve_device


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--biases", type=int, nargs="+", default=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    ap.add_argument("--modes", nargs="+", default=["paper", "subnormal"],
                    choices=["paper", "legacy", "subnormal"])
    ap.add_argument("--no-accuracy", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    model, ckpt = load_model(args.checkpoint)
    orig = conv_linear_weights(model)
    absw = np.concatenate([np.abs(w).reshape(-1) for w in orig.values()]).astype(np.float64)
    loader = None
    if not args.no_accuracy:
        norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
        loader = fdata.get_test_loader(cfg.data_path(), norm, cfg.eval_batch_size, cfg.num_workers,
                                       cfg.test_subset, device.type == "cuda")
        fp32 = evaluate(model, loader, device)["accuracy_percent"]
        print(f"FP32 accuracy: {fp32:.2f}%   max |w| = {absw.max():.4f}")
    rows = []
    hdr = f"{'mode':10s} {'bias':>4s} {'max repr':>10s} {'%clipped':>9s} {'%flushed':>9s} {'SQNR dB':>8s} {'acc %':>7s}"
    print(hdr)
    for mode in args.modes:
        for b in args.biases:
            q, _ = quantize_model(model, mode=mode, bias=b)
            r = sqnr_report(orig, conv_linear_weights(q))
            vmax = e4m11.max_finite(b)
            clipped = 100.0 * float((absw > vmax).mean())
            flushed = 0.0 if mode == "subnormal" else 100.0 * float((absw < 2.0 ** -b).mean())
            acc = None if loader is None else evaluate(q, loader, device)["accuracy_percent"]
            rows.append({"mode": mode, "bias": b, "max_representable": vmax, "pct_clipped": clipped,
                         "pct_flushed": flushed, "sqnr_db": r["overall"]["sqnr_db"], "accuracy": acc})
            print(f"{mode:10s} {b:4d} {vmax:10.4g} {clipped:9.4f} {flushed:9.2f} "
                  f"{r['overall']['sqnr_db']:8.2f} {'' if acc is None else f'{acc:7.2f}'}")
    out = cfg.results_dir() / "bias_sweep.json"
    dump_json({"checkpoint": args.checkpoint, "abs_max_weight": float(absw.max()), "rows": rows}, out)
    print("written:", out)


if __name__ == "__main__":
    main()
