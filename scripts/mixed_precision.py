#!/usr/bin/env python
"""Step 3b/3c: choose which layers move from E4M11 (16 bit) to an 8-bit format and measure clean metrics.

Phase `sweep` (default): for every 8-bit format (arm), size target and scorer, select the layers with the
Fisher/error scores (`fr.allocation`), quantize and evaluate on the VALIDATION split only. Resumable.
Phase `final`: evaluate the saved selections on the TEST split (lambda chosen on validation) and write
`mixed_report.md`. Phase `churn`: prediction flips / KL of every saved selection vs the all-E4M11 model
(validation for all configurations, test for the same ones `final` evaluates) -> `mixed_churn_report.md`.
Run `scripts/fisher_sensitivity.py` first.

  python scripts/mixed_precision.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
  python scripts/mixed_precision.py --checkpoint ... --run-name scratch --phase final

Arms: int8_pt (INT8 per-tensor, paper Eq. 3), int8_pc (per-channel ablation), fp8 (E4M3 with subnormals, no
scale), fp8_scaled (E4M3 + per-tensor power-of-two scale). Targets are average payload bits per weight
(16 = all E4M11, 8 = all 8-bit).
"""
import argparse
import logging
from pathlib import Path

import _common  # noqa: F401
from fr import allocation
from fr.config import add_config_args, config_from_args
from fr.formats8 import FORMATS8
from fr.mixed_run import DEFAULT_LAMBDAS, run_churn, run_final, run_sweep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--phase", choices=["sweep", "final", "churn"], default="sweep")
    ap.add_argument("--fisher-kind", choices=["true", "empirical"], default="true")
    ap.add_argument("--fisher-file", default=None, help="default: results/<run>/sensitivity/fisher_<kind>.pt")
    ap.add_argument("--arms", nargs="+", choices=FORMATS8, default=list(FORMATS8))
    ap.add_argument("--targets", type=float, nargs="+", default=[14.0, 12.0, 10.0])
    ap.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    ap.add_argument("--random-draws", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="feasible window: switched weights in [need, need*(1+tol)] (doubled if empty)")
    ap.add_argument("--score-norm", choices=allocation.NORMS, default="rank")
    ap.add_argument("--fisher-mode", choices=allocation.FISHER_MODES, default="per_param")
    ap.add_argument("--e4m11-mode", choices=["paper", "legacy", "subnormal"], default="subnormal")
    ap.add_argument("--e4m11-bias", type=int, default=7)
    ap.add_argument("--val-subset", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    out_dir = cfg.results_dir() / "mixed"

    if args.phase == "sweep":
        fisher = Path(args.fisher_file) if args.fisher_file else \
            cfg.results_dir() / "sensitivity" / f"fisher_{args.fisher_kind}.pt"
        if not fisher.exists():
            raise SystemExit(f"{fisher} not found; run scripts/fisher_sensitivity.py first")
        res = run_sweep(cfg, args.checkpoint, fisher, args.arms, args.targets, args.lambdas,
                        args.random_draws, args.tol, args.score_norm, args.fisher_mode,
                        args.e4m11_mode, args.e4m11_bias, args.val_subset, out_dir)
        print("validation rows:", res["csv"])
    elif args.phase == "churn":
        res = run_churn(cfg, args.checkpoint, out_dir)
        print(Path(res["report"]).read_text(encoding="utf-8"))
        print("written:", res["report"])
    else:
        res = run_final(cfg, args.checkpoint, out_dir)
        print(Path(res["report"]).read_text(encoding="utf-8"))
        print("written:", res["report"])


if __name__ == "__main__":
    main()
