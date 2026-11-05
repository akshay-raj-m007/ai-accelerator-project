#!/usr/bin/env python
"""FP32 test accuracy + weight statistics (paper Table 3 style).

  python scripts/eval_fp32.py --checkpoint checkpoints/finetune/best.pth --run-name finetune
  python scripts/eval_fp32.py --checkpoint checkpoints/resnet18_cifar10_checkpoint.pth \
      --run-name legacy --diagnose-normalization --test-subset 1000
"""
import argparse
import logging

import _common  # noqa: F401
from fr.config import add_config_args, config_from_args
from fr.step1 import eval_fp32


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--diagnose-normalization", action="store_true",
                    help="evaluate with and without CIFAR mean/std (diagnostic only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    out = eval_fp32(cfg, args.checkpoint, args.diagnose_normalization)
    print({k: v for k, v in out.items() if k in ("metrics", "normalization_diagnostic")})
    print("weight stats:", out["weight_stats"])
    print("written:", cfg.results_dir() / "fp32_report.json")


if __name__ == "__main__":
    main()
