#!/usr/bin/env python
"""Step 3a: diagonal Fisher information of every Conv2d/Linear weight (LAMPQ-style sensitivity).

Uses `--calib-size` un-augmented images from the TRAINING part of the split (default) or, with `--source val`,
the first images of the held-out VALIDATION split (the sweep must then skip them: `--val-skip`). Never the test set.
Writes results/<run>/sensitivity/fisher_{true,empirical}.pt, fisher_*_layers.json and sensitivity_meta.json.

  python scripts/fisher_sensitivity.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
"""
import argparse
import logging

import _common  # noqa: F401
from fr.config import add_config_args, config_from_args
from fr.mixed_run import run_fisher
from fr.sensitivity import KINDS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS),
                    help="true = labels sampled from the model (LAMPQ Lemma 2); empirical = dataset labels")
    ap.add_argument("--calib-size", type=int, default=1024, help="LAMPQ uses 1,024")
    ap.add_argument("--source", choices=["train", "val"], default="train",
                    help="train = first images of the training split (default, memorised by the network); "
                         "val = first images of the held-out validation split (held-out Fisher)")
    ap.add_argument("--out-name", default=None,
                    help="results/<run>/<out-name>; default sensitivity (train) or sensitivity_heldout (val)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    out = cfg.results_dir() / (args.out_name or ("sensitivity" if args.source == "train" else "sensitivity_heldout"))
    run_fisher(cfg, args.checkpoint, args.kinds, args.calib_size, out, args.source)
    print("written:", out)


if __name__ == "__main__":
    main()
