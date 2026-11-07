#!/usr/bin/env python
"""Part D: random bit-flip campaigns on MIXED E4M11 / 8-bit storage, compared across storage layouts.

A *layout* is one assignment of formats to layers:
  e4m11                          every layer E4M11 (the baseline, 16 bit/weight)
  all:<arm>                      every layer in an 8-bit format (arm = int8_pt | int8_pc | fp8 | fp8_scaled | int4_pt | int4_pc | fp4_scaled)
  <arm>:T12:product              a saved selection of the validation sweep (results/<run>/mixed/selections.json):
  <arm>:T12:lambda:0             <arm>:T<bits>:<method>[:<param>], method = lambda (param = lambda value),
  <arm>:T12:random:0             product, worst, random (param = draw)

Every layout gets its own resumable campaign directory results/<run>/mixed_faults/<sampling>/<layout>/ and a
comparison table results/<run>/mixed_faults/<sampling>/comparison.md. Protocol, bit groups and sampling models:
src/fr/faults_mixed.py (docstring). Default: group `any` (every bit of every word), k = 8, 100 trials, sampling
`weights` (k weights uniformly, then a bit uniformly; the paper's wording). In that setting the struck weights
are the same for every layout, so layouts are compared pairwise.

  # core comparison: the baseline, the all-8-bit models and the saved selections at 12 bits/weight
  python scripts/mixed_fault_campaign.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch

  # decomposition by bit group (paper groups) and the physical bit-uniform sampling
  python scripts/mixed_fault_campaign.py --checkpoint ... --run-name scratch \\
      --groups exponent sign mantissa_msb mantissa_lsb --ks 8 --samplings weights bits
"""
import argparse
import json
import logging
from pathlib import Path

import _common  # noqa: F401
import torch
from fr import data as fdata
from fr import faults, faults_mixed as FM, mixed
from fr.config import add_config_args, config_from_args
from fr.formats8 import ALL_LOW
from fr.model import load_model
from fr.quantize import conv_linear_weights
from fr.utils import dump_json, env_info, resolve_device

DEFAULT_ARMS = ["int8_pc", "int8_pt", "fp8_scaled"]
DEFAULT_METHODS = ["product", "lambda:0", "random:0", "random:1", "worst"]


def build_specs(args) -> list:
    specs = list(args.layouts or [])
    if not args.layouts:
        if not args.no_refs:
            specs += ["e4m11"] + [f"all:{a}" for a in args.arms]
        for arm in args.arms:
            for t in args.targets:
                for m in args.methods:
                    specs.append(f"{arm}:T{t:g}:{m}")
    seen, out = set(), []
    for s in specs:
        slug = FM.normalize_spec(s)[0]
        if slug not in seen:
            seen.add(slug)
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sweep-dir", default=None, help="default: results/<run>/mixed")
    ap.add_argument("--out-dir", default=None, help="default: results/<run>/mixed_faults")
    ap.add_argument("--layouts", nargs="+", default=None, help="explicit layout specs (see above)")
    ap.add_argument("--arms", nargs="+", choices=ALL_LOW, default=DEFAULT_ARMS)
    ap.add_argument("--targets", type=float, nargs="+", default=[12.0])
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--no-refs", action="store_true", help="skip e4m11 and all-8-bit reference layouts")
    ap.add_argument("--groups", nargs="+", default=["any"],
                    help="any | exponent sign mantissa_msb mantissa_lsb | bitN")
    ap.add_argument("--ks", type=int, nargs="+", default=[8])
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--samplings", nargs="+", choices=FM.SAMPLINGS, default=["weights"])
    ap.add_argument("--layer", default=None, help="restrict faults to one layer")
    ap.add_argument("--e4m11-mode", choices=["paper", "legacy", "subnormal"], default=None,
                    help="default: the mode of the sweep that produced the selections")
    ap.add_argument("--e4m11-bias", type=int, default=None)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the number of evaluations")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else cfg.results_dir() / "mixed"
    out_root = Path(args.out_dir) if args.out_dir else cfg.results_dir() / "mixed_faults"

    sweep_meta = json.load(open(sweep_dir / "sweep_meta.json")) if (sweep_dir / "sweep_meta.json").exists() else {}
    mode = args.e4m11_mode or sweep_meta.get("e4m11_mode", "subnormal")
    bias = args.e4m11_bias if args.e4m11_bias is not None else sweep_meta.get("bias", 7)
    sel_path = sweep_dir / "selections.json"
    selections = json.load(open(sel_path)) if sel_path.exists() else {}

    specs = build_specs(args)
    n_evals = len(specs) * len(args.groups) * len(args.ks) * args.trials * len(args.samplings)
    print(f"{len(specs)} layouts x {len(args.groups)} group(s) x {len(args.ks)} k x {args.trials} trials x "
          f"{len(args.samplings)} sampling(s) = {n_evals} full-test-set evaluations (about 2-3 s each on an RTX 3050)")
    for s in specs:
        print("  ", s)
    if args.dry_run:
        return

    if not args.summarize_only:
        model, ckpt = load_model(args.checkpoint)
        norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
        loader = fdata.get_test_loader(cfg.data_path(), norm, cfg.eval_batch_size, cfg.num_workers,
                                       cfg.test_subset, False)
        X, y = faults.load_tensors(loader, device)
        names = list(conv_linear_weights(model))
        qcache = {}
        for sampling in args.samplings:
            for spec in specs:
                slug, assignment, label = FM.resolve_assignment(spec, selections, names)
                qmodel, buffers = mixed.quantize_mixed(model, assignment, mode, bias, cache=qcache)
                inj = FM.MixedFaultInjector(qmodel, buffers, mode, bias, "clamp", device)
                assert inj.clean_weights_unchanged(), "clean decode does not match the model weights"
                acc = mixed.bit_accounting(buffers)
                print(f"\n=== {slug} [{label}]  {acc['avg_bits_payload']:.2f} bits/weight, "
                      f"{sum(1 for b in buffers.values() if b['format'] != 'e4m11')} layers in low-bit; "
                      f"sampling={sampling}")
                meta = {"layout": spec, "label": label, "layers8": sorted(k for k, a in assignment.items()),
                        "avg_bits_payload": acc["avg_bits_payload"],
                        "avg_bits_with_metadata": acc["avg_bits_with_metadata"],
                        "weights_by_format": acc["weights_by_format"], "checkpoint": str(args.checkpoint),
                        "env": env_info(), "normalization": fdata.normalization_to_config(norm)}
                FM.run_mixed_campaign(inj, X, y, out_root / sampling / slug, args.groups, args.ks, args.trials,
                                      cfg.seed, sampling, args.layer, meta)
                assert inj.clean_weights_unchanged(), "weights were not restored"
                del inj, qmodel
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    for sampling in args.samplings:
        d = out_root / sampling
        camps = FM.collect(sorted(p for p in d.iterdir() if p.is_dir())) if d.exists() else []
        md = FM.render_comparison(camps, reference="e4m11")
        (d / "comparison.md").parent.mkdir(parents=True, exist_ok=True)
        (d / "comparison.md").write_text(md, encoding="utf-8")
        dump_json({c["slug"]: FM.summarize_trials(c["rows"], c["meta"]["clean"]["accuracy_percent"],
                                                  next((x["meta"]["clean"]["accuracy_percent"] for x in camps
                                                        if x["slug"] == "e4m11"),
                                                       c["meta"]["clean"]["accuracy_percent"]))
                   for c in camps}, d / "comparison.json")
        print(md)
        print("written:", d / "comparison.md")


if __name__ == "__main__":
    main()
