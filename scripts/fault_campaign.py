#!/usr/bin/env python
"""Random bit-flip campaign on the packed E4M11 weights (paper Sec. 4 / App. B.3).

Default = the fixed baseline: E4M11 with subnormals, bias 7, Conv/Linear weights. The fault seed is
--seed (Config.seed). Output: results/<run>/faults_<mode>_b<bias>[_<layer>]/{trials.csv,
campaign_meta.json, summary.json, summary.md}. Re-running the same command resumes.

  # paper protocol (4 groups x k=1..8 x 100 trials, whole weight buffer)
  python scripts/fault_campaign.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch

  # Day-9 style: 100 flips of bit 14 in one layer
  python scripts/fault_campaign.py --checkpoint ... --run-name scratch --groups bit14 --ks 100 \
      --layer layer4.0.conv1 --trials 3
"""
import argparse
import json
import logging
from pathlib import Path

import _common  # noqa: F401
from fr import data as fdata
from fr import faults
from fr.config import add_config_args, config_from_args
from fr.model import load_model
from fr.quantize import quantize_model
from fr.report import load_targets
from fr.utils import dump_json, env_info, resolve_device

DEFAULT_TARGETS = Path(__file__).resolve().parents[1] / "configs" / "paper_targets.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--e4m11-mode", choices=["paper", "legacy", "subnormal"], default="subnormal")
    ap.add_argument("--e4m11-bias", type=int, default=7)
    ap.add_argument("--all-ones", choices=["clamp", "ieee", "literal"], default="clamp")
    ap.add_argument("--groups", nargs="+", default=list(faults.PAPER_GROUPS),
                    help="exponent sign mantissa_msb mantissa_lsb | any | bitN")
    ap.add_argument("--ks", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--layer", default=None, help="restrict faults to one layer (paper: whole buffer)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--targets", default=str(DEFAULT_TARGETS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)

    tag = f"faults_{args.e4m11_mode}_b{args.e4m11_bias}" + (f"_{args.layer}" if args.layer else "")
    out_dir = cfg.results_dir() / tag

    model, ckpt = load_model(args.checkpoint)
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    loader = fdata.get_test_loader(cfg.data_path(), norm, cfg.eval_batch_size, cfg.num_workers,
                                   cfg.test_subset, False)
    X, y = faults.load_tensors(loader, device)
    fp32 = faults.eval_tensors(model.to(device), X, y)
    print(f"FP32 accuracy on {X.shape[0]} test images: {fp32['accuracy_percent']:.2f}%")

    qmodel, buffers = quantize_model(model.cpu(), mode=args.e4m11_mode, scope="convlinear",
                                     all_ones=args.all_ones, bias=args.e4m11_bias)
    inj = faults.FaultInjector(qmodel, buffers, args.e4m11_mode, args.e4m11_bias, args.all_ones, device)
    assert inj.clean_weights_unchanged()
    print(f"E4M11 buffer: {inj.total:,} words in {len(buffers)} tensors")

    csv_path = out_dir / "trials.csv"
    if not args.summarize_only:
        faults.run_campaign(inj, X, y, out_dir, args.groups, args.ks, args.trials, cfg.seed, args.layer,
                            meta={"checkpoint": args.checkpoint, "fp32": fp32, "env": env_info(),
                                  "normalization": fdata.normalization_to_config(norm)})
        assert inj.clean_weights_unchanged(), "weights were not restored"
    meta = json.load(open(out_dir / "campaign_meta.json"))
    summary = faults.summarize(csv_path, meta["clean"]["accuracy_percent"], fp32["accuracy_percent"])
    targets = load_targets(args.targets).get("fault_drop_pp_at_8_flips") if args.layer is None else None
    dump_json({"meta": meta, "summary": summary}, out_dir / "summary.json")
    md = faults.render_summary_md(summary, meta, fp32["accuracy_percent"], targets)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print("written:", out_dir)


if __name__ == "__main__":
    main()
