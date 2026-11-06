#!/usr/bin/env python
"""FP32 -> E4M11 post-training quantization: accuracy, SQNR, paper comparison.

Writes results/<run_name>/step1_report.json and step1_report.md

  python scripts/quantize_eval.py --checkpoint checkpoints/finetune/best.pth --run-name finetune --compare-modes
  # reproduce the old notebook (all parameters, no carry):
  python scripts/quantize_eval.py --checkpoint checkpoints/resnet18_cifar10_checkpoint.pth \
      --run-name legacy --e4m11-mode legacy --quantize-scope all --normalize none
"""
import argparse
import logging
from pathlib import Path

import _common  # noqa: F401
from fr.config import add_config_args, config_from_args
from fr.step1 import quantize_eval

DEFAULT_TARGETS = Path(__file__).resolve().parents[1] / "configs" / "paper_targets.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--e4m11-mode", choices=["paper", "legacy", "subnormal"], default="paper")
    ap.add_argument("--compare-modes", action="store_true", help="run paper, legacy and subnormal")
    ap.add_argument("--quantize-scope", choices=["convlinear", "all"], default="convlinear")
    ap.add_argument("--all-ones", choices=["clamp", "ieee", "literal"], default="clamp",
                    help="decode policy for the reserved all-ones exponent (paper mode only)")
    ap.add_argument("--targets", default=str(DEFAULT_TARGETS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    modes = ("paper", "legacy", "subnormal") if args.compare_modes else (args.e4m11_mode,)
    quantize_eval(cfg, args.checkpoint, modes, args.quantize_scope, args.all_ones, Path(args.targets))
    print("written:", cfg.results_dir() / "step1_report.md")


if __name__ == "__main__":
    main()
