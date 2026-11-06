#!/usr/bin/env python
"""How stable are the Fisher-based layer rankings? (subset / sample-size stability, concentration of the trace
on a few images, stability of the Fisher-only layer selection, training vs held-out images)

Computes per-sample layer traces on a pool of images from each source (training split and validation split),
caches them in results/<run>/sensitivity/traces_<source>_<kind>.npz and writes fisher_stability.md/.json.

  python scripts/fisher_stability.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
"""
import argparse
import logging
from pathlib import Path

import _common  # noqa: F401
from fr.config import add_config_args, config_from_args
from fr.mixed_run import run_stability
from fr.sensitivity import KINDS
from fr.stability import DEFAULT_SIZES


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pool", type=int, default=4096, help="images per source")
    ap.add_argument("--sources", nargs="+", choices=["train", "val"], default=["train", "val"])
    ap.add_argument("--kind", choices=KINDS, default="true")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    res = run_stability(cfg, args.checkpoint, args.sources, args.pool, args.kind, args.sizes)
    print(Path(res["report"]).read_text(encoding="utf-8"))
    print("written:", res["report"])


if __name__ == "__main__":
    main()
