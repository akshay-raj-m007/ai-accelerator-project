#!/usr/bin/env python
"""MobileNetV2 protocol Section 6, input R_scale: exhaustive single-bit faults of every INT layer's float32 SCALE word (GPU machine).

Section 4 faulted only the weight codes. A flipped bit of the float32 scale of an INT layer rescales every weight of the layer, so it is a second, separate
fault process (never merged into R). For every INT format (int8_pt, int4_pt) and every layer, ALL 32 bits of the scale word are flipped one at a time in the
stored bytes (exhaustive, no sampling), the layer is decoded with the faulty scale, and the model is evaluated on the same fixed class-stratified 1,000-image validation
subset as Section 4 screening. The result per (layer, format) is the mean accuracy drop over the 32 bits = the expected drop of ONE uniformly drawn scale-bit flip.

  python scripts/stage6_scale_faults.py --checkpoint $CK --run-name mobilenetv2_cifar10 --out-dir reports/mobilenetv2_cifar10/stage_06_optimization/scale_faults

Writes scale_trials.jsonl (resumable, one record per layer x format x bit with the Section 5.1 fields and hashes), scale_summary.csv (per layer x format: mean / median /
worst drop, drops by field, non-finite failures), scale_format_summary.csv, scale_manifest.json. Validation images only; the test set is never read.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from fr import data as fdata
from fr import stage2 as S2
from fr import stage3 as S3
from fr import stage4 as S4
from fr.baseline_profile import sha256_file, sha256_ints
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, env_info, git_info, resolve_device
from stage4_encoded_fault_injection import stratified_subset

EXPERIMENT = "jfamp-mobilenetv2-cifar10-stage6-scale-word-faults"
INT_FORMATS = ["int8_pt", "int4_pt"]
SUMMARY_FIELDS = ["layer_index", "layer_name", "format_key", "n_bits", "clean_accuracy", "mean_drop_pp", "median_drop_pp", "worst_drop_pp", "p90_drop_pp",
                  "mean_drop_pp_sign", "mean_drop_pp_exponent", "mean_drop_pp_fraction", "failures", "failure_rate", "n_restored_exactly", "sample_count"]


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def summarise(records, module_names):
    by = {}
    for r in records:
        by.setdefault((r["layer_id"], r["format"]), []).append(r)
    rows = []
    for (l, fmt), rs in sorted(by.items()):
        d = np.array([r["clean_accuracy"] - r["fault_accuracy"] for r in rs])
        def grp(name):
            v = [r["clean_accuracy"] - r["fault_accuracy"] for r in rs if r["flips"][0]["field"] == name]
            return float(np.mean(v)) if v else float("nan")
        rows.append({"layer_index": l, "layer_name": module_names[l], "format_key": fmt, "n_bits": len(rs), "clean_accuracy": rs[0]["clean_accuracy"],
                     "mean_drop_pp": float(d.mean()), "median_drop_pp": float(np.median(d)), "worst_drop_pp": float(d.max()), "p90_drop_pp": float(np.percentile(d, 90)),
                     "mean_drop_pp_sign": grp("sign"), "mean_drop_pp_exponent": grp("exponent"), "mean_drop_pp_fraction": grp("fraction"),
                     "failures": int(sum(r["failure"] for r in rs)), "failure_rate": float(np.mean([r["failure"] for r in rs])),
                     "n_restored_exactly": int(sum(r["restored_exactly"] for r in rs)), "sample_count": rs[0]["sample_count"]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--formats", nargs="+", default=INT_FORMATS, choices=INT_FORMATS)
    ap.add_argument("--subset-size", type=int, default=1000, help="the Section 4 declared class-stratified validation subset (0 = full 5,000)")
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--only-layers", nargs="*", default=None, help="smoke runs only")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    out = Path(args.out_dir) if args.out_dir else base / "stage_06_optimization" / "scale_faults"
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(args.checkpoint)
    man = json.load(open(qdir / "quantizer_manifest.json"))
    ck_sha = sha256_file(ck)
    if man["checkpoint"]["sha256"] != ck_sha:
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    frozen = man["frozen_config_sha256"]
    model, ckpt = load_model(ck)
    model.eval()
    mods = quantizable_modules(model)
    module_names = [n for n, _ in mods]
    layer_names = [f"{n}.weight" for n in module_names]
    codecs = S2.codec_registry()
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    X, y = load_tensors(fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=None), device)
    sel = stratified_subset(y.cpu().numpy(), args.subset_size, args.eval_seed)
    idx = torch.from_numpy(sel).to(device)
    X, y = X[idx], y[idx]
    subset = {"name": "full-validation" if len(sel) == 5000 else f"stratified-{len(sel)}-seed{args.eval_seed}", "images": int(y.numel()),
              "positions_in_validation_set_sha256": sha256_ints(sel), "seed": args.eval_seed}
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    commit = git_info()["commit"]
    layers = list(range(len(module_names))) if not args.only_layers else [module_names.index(n) for n in args.only_layers]
    log = S4.TrialLog(out / "scale_trials.jsonl")
    if log.duplicates:
        sys.exit("duplicate trial ids in scale_trials.jsonl")
    t0, n_new = time.time(), 0
    for fmt in args.formats:
        stored = S2.load_encoded(qdir / man["formats"][fmt]["encoded_file"], layer_names)
        if S2.content_sha256(stored) != man["formats"][fmt]["content_sha256"]:
            sys.exit(f"{fmt}: encoded weights differ from the frozen content hash")
        store = S3.EncodedStore(fmt, stored)
        inj = S3.EncodedFaultInjector(model, store, codecs[fmt], device)
        clean = S3.evaluate(inj.model, X, y)
        ctx = {"experiment_id": EXPERIMENT, "run_id": run_id, "model": "mobilenetv2_cifar (width 1.0, stride-1 stem, 10 classes)",
               "dataset": f"CIFAR-10 validation holdout, {subset['name']}", "checkpoint_hash": ck_sha,
               "codec_version": f"{fmt}|stage2-frozen:{frozen[:12]}|{S3.INJECTOR_VERSION}|git:{commit[:8]}"}
        for l in layers:
            for bit in range(32):
                tid = f"{fmt}|scales|{module_names[l]}|bit{bit}"
                if log.has(tid):
                    continue
                rec = S3.run_trial(inj, X, y, clean, ctx, fmt=fmt, scope_layers=[l], scope_name=module_names[l], scope_kind="scales", group="all", k=1, trial=bit,
                                   base_seed=args.eval_seed, explicit_scale_flips=[(l, bit)])
                if not rec["restored_exactly"] or not rec["flip_applied_to_encoded_bytes"]:
                    sys.exit(f"{tid}: restoration / flip check failed")
                rec.update({"trial_id": tid, "campaign": "scale_exhaustive", "eval_subset": subset["name"], "schema": "mnv2-stage6-scale-trial-v1"})
                log.append(rec)
                n_new += 1
            print(f"  {fmt} {module_names[l]:24s} done ({time.time() - t0:.0f} s)", flush=True)
        # the injected store must be back to the frozen content
        assert S2.content_sha256(store.to_encoded()) == man["formats"][fmt]["content_sha256"], "stored bytes changed during the campaign"
        del inj
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log.close()
    recs = [r for r in S4.load_records(out / "scale_trials.jsonl") if r["format"] in args.formats]
    rows = summarise(recs, module_names)
    expect = len(args.formats) * len(layers) * 32
    if len(recs) != expect and not args.only_layers:
        sys.exit(f"{len(recs)} records, expected {expect}")
    write_csv(out / "scale_summary.csv", rows, SUMMARY_FIELDS)
    fmt_rows = []
    for fmt in args.formats:
        d = np.array([r["mean_drop_pp"] for r in rows if r["format_key"] == fmt])
        fmt_rows.append({"format_key": fmt, "n_layers": int(d.size), "mean_of_layer_means_pp": float(d.mean()), "worst_layer_mean_pp": float(d.max()),
                         "layers_with_mean_drop_gt_1pp": int((d > 1.0).sum()), "layers_with_mean_drop_gt_0p5pp": int((d > 0.5).sum())})
    write_csv(out / "scale_format_summary.csv", fmt_rows)
    dump_json({"stage": "MobileNetV2 Section 6 input: exhaustive scale-word faults", "experiment_id": EXPERIMENT, "run_id": run_id, "checkpoint_sha256": ck_sha,
               "stage2_frozen_config_sha256": frozen, "eval_subset": subset, "formats": args.formats, "layers": len(layers), "bits_per_scale_word": 32,
               "trials_recorded": len(recs), "all_restored_exactly": all(r["restored_exactly"] for r in recs), "git": git_info(), "environment": env_info(),
               "seconds": time.time() - t0, "records_sha256": hashlib.sha256(open(out / "scale_trials.jsonl", "rb").read()).hexdigest()}, out / "scale_manifest.json")
    print(f"written: {out}  ({len(recs)} records, {n_new} new)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
