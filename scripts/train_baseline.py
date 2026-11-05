#!/usr/bin/env python
"""Train from scratch or fine-tune the legacy checkpoint.

Examples
  python scripts/train_baseline.py --config configs/finetune.yaml --run-name finetune
  python scripts/train_baseline.py --config configs/scratch.yaml  --run-name scratch
  python scripts/train_baseline.py --config configs/scratch.yaml  --run-name scratch --resume
"""
import argparse

import _common  # noqa: F401
from fr.config import add_config_args, config_from_args
from fr.train import train


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    cfg = config_from_args(ap.parse_args())
    train(cfg)


if __name__ == "__main__":
    main()
