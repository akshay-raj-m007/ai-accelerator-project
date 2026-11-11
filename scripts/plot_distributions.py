#!/usr/bin/env python
"""Per-layer weight and activation distribution plots, 8 plots per page, one heading and legend per plot.

  python scripts/plot_distributions.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch

Writes under results/<run>/jfamp/figures/:
  weights_distribution.pdf            + weights_distribution_pageN.png        (21 layers -> 3 pages)
  activations_input_distribution.pdf  + activations_input_distribution_pageN.png
  (with --act-point output also activations_output_*)  and distribution_stats.json (all plotted statistics)

Activations are measured on the first --n-images images of the VALIDATION holdout (never the test set), FP32, eval mode.
Default activation point = the INPUT of each Conv2d/Linear layer; --act-point output uses the layer output (pre-BatchNorm).
Needs matplotlib (pip install matplotlib).
"""
import argparse
import logging
from pathlib import Path

import _common  # noqa: F401
import torch
from fr import data as fdata
from fr import dist_plots as DP
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model
from fr.utils import dump_json, resolve_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default=None, help="default results/<run>/jfamp/figures")
    ap.add_argument("--n-images", type=int, default=512, help="validation images used for the activations")
    ap.add_argument("--act-point", choices=["input", "output", "both"], default="input")
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--max-samples", type=int, default=2_000_000, help="activation values kept per layer for the histogram")
    ap.add_argument("--skip-activations", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    out = Path(args.out_dir) if args.out_dir else cfg.results_dir() / "jfamp" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(args.checkpoint)
    model.to(device).eval()
    tag = f"ResNet-18 / CIFAR-10 (FP32 checkpoint {Path(args.checkpoint).name})"
    summary = {}

    ws = DP.weight_stats(model)
    DP.plot_pages(ws, "weights", f"Weight distribution per layer - {tag}", out / "weights_distribution.pdf",
                  out / "weights_distribution", args.bins,
                  "all weights of the Conv2d/Linear tensor; y axis log scale; dashed = mean ± 1 std, dotted = p0.1 / p99.9")
    summary["weights"] = {n: {k: v for k, v in s.items() if k != "values"} for n, s in ws.items()}
    print("weights: written", out / "weights_distribution.pdf")

    if not args.skip_activations:
        norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
        val_size, split_seed = split_params(ckpt, cfg)
        loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed,
                                      cfg.num_workers, False, subset=args.n_images)
        X, _ = load_tensors(loader, device)
        points = ["input", "output"] if args.act_point == "both" else [args.act_point]
        for pt in points:
            st = DP.activation_stats(model, X, pt, max_samples=args.max_samples)
            what = "input of" if pt == "input" else "output of"
            DP.plot_pages(st, "activations", f"Activation distribution per layer ({what} each Conv2d/Linear) - {tag}",
                          out / f"activations_{pt}_distribution.pdf", out / f"activations_{pt}_distribution", args.bins,
                          f"{X.shape[0]} validation images, FP32; exact mean/std/min/max/zeros over all values, "
                          f"histogram + percentiles from a random subsample; y axis log scale")
            summary[f"activations_{pt}"] = {n: {k: v for k, v in s.items() if k != "values"} for n, s in st.items()}
            print(f"activations ({pt}): written", out / f"activations_{pt}_distribution.pdf")
    summary["activation_images"] = None if args.skip_activations else int(X.shape[0])
    dump_json(summary, out / "distribution_stats.json")
    print("written:", out)


if __name__ == "__main__":
    main()
