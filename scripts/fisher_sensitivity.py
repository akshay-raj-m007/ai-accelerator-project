#!/usr/bin/env python
"""Step 3a: diagonal Fisher information of every Conv2d/Linear weight (LAMPQ-style sensitivity).

Uses `--calib-size` un-augmented images from the TRAINING part of the split (never validation or test).
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
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    run_fisher(cfg, args.checkpoint, args.kinds, args.calib_size)
    print("written:", cfg.results_dir() / "sensitivity")


if __name__ == "__main__":
    main()
