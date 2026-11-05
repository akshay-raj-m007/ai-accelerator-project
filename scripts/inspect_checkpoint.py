#!/usr/bin/env python
"""Print the metadata of a checkpoint (notably ``config``: transforms/normalization)."""
import argparse
import pprint

import _common  # noqa: F401
from fr.model import load_checkpoint
from fr.data import norm_from_checkpoint_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    args = ap.parse_args()
    ck = load_checkpoint(args.checkpoint)
    print("keys:", list(ck))
    for k, v in ck.items():
        if k in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                 "scaler_state_dict", "rng_state"):
            continue
        print(f"\n[{k}]")
        pprint.pprint(v, width=100)
    print("\nnormalization inferred from config:", norm_from_checkpoint_config(ck.get("config")))


if __name__ == "__main__":
    main()
