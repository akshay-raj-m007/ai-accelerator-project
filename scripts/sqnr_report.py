#!/usr/bin/env python
"""Weight-space SQNR of the trained model, after quantization (uniform formats and the final assignment) and after bit flips (CPU only, a few minutes).

Works for the ResNet-18 run and the MobileNetV2 run (same code):

  # MobileNetV2 (faults drawn through the S6 protection model)
  python scripts/sqnr_report.py --checkpoint $FR_ROOT/checkpoints/mobilenetv2_cifar10/best.pth --run-name mobilenetv2_cifar10 \
      --opt-dir reports/mobilenetv2_cifar10/stage_06_fisher --out-dir reports/mobilenetv2_cifar10/sqnr --draw s6
  # ResNet-18 (Stage 5 draws; run name scratch; its e2e results live in reports/scratch/jfamp/optimization)
  python scripts/sqnr_report.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch \
      --opt-dir reports/scratch/jfamp/optimization --out-dir reports/scratch/jfamp/sqnr --draw e2

SQNR = 10 log10( sum ||w||^2 / sum ||w - w_hat||^2 ) over all quantized Conv/Linear weights, w = FP32 checkpoint (see src/fr/sqnr.py).
Rows: FP32 (reference, error 0, SQNR = +inf), every uniform unprotected format, the uniform protected references, and the frozen final assignment.
After-fault SQNR uses the SAME fault draws as the accuracy trials of the end-to-end run (same seeds); each regenerated trial is compared with the recorded one
(flips identical?) and joined with its recorded accuracy drop, so SQNR can be plotted against accuracy loss trial by trial.
Writes sqnr_summary.csv, sqnr_trials.csv, sqnr_report.json.
"""
import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from fr import stage2 as S2
from fr import stage3 as S3
from fr import stage5_e2e as E2
from fr import stage6 as S6
from fr import weight_sqnr as Q
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, git_info


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def options_of(item):
    """Per-layer option tokens of an end-to-end assignment record (MobileNetV2 stores 'options', ResNet-18 'datatype_per_layer' + 'protection_per_layer')."""
    if "options" in item:
        return list(item["options"])
    prot = item["protection_per_layer"]
    return [dt if prot[l] in ("none", None, "") else f"{dt}+{prot[l]}" for l, dt in item["datatype_per_layer"].items()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None, help="default results/<run>/jfamp/quantization (frozen Stage 2 encodings)")
    ap.add_argument("--opt-dir", required=True, help="folder with assignment.json, end_to_end_assignment_metrics.json, end_to_end_fault_trials.jsonl, end_to_end_manifest.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--draw", choices=["s6", "e2"], required=True, help="s6 = MobileNetV2 run (ProtectedFaults), e2 = ResNet-18 run (Stage 5 draws)")
    ap.add_argument("--distributions", nargs="+", default=["weights_k8", "scales_k1"])
    ap.add_argument("--base-seed", type=int, default=None, help="default: the base seed of the end-to-end manifest")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    odir, out = Path(args.opt_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qdir = Path(args.quant_dir) if args.quant_dir else cfg.results_dir() / "jfamp" / "quantization"
    ck = Path(args.checkpoint)
    man2 = json.load(open(qdir / "quantizer_manifest.json"))
    if man2["checkpoint"]["sha256"] != sha256_file(ck):
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    metrics = json.load(open(odir / "end_to_end_assignment_metrics.json"))
    final = json.load(open(odir / "assignment.json"))
    e2e_manifest = json.load(open(odir / "end_to_end_manifest.json"))
    base_seed = args.base_seed if args.base_seed is not None else int(e2e_manifest["settings"]["base_seed"])
    items = OrderedDict((a["assignment_id"], a) for a in metrics["assignments"])
    ids = [i for i in items if i.startswith("U_")] + [final["assignment_id"]]
    recorded = {}
    with open(odir / "end_to_end_fault_trials.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["assignment_id"] in ids:
                recorded[r["trial_id"]] = r
    model, _ = load_model(ck)
    model.eval()
    mods = quantizable_modules(model)
    fp32_w = [m.weight for _, m in mods]
    lnames = [f"{n}.weight" for n, _ in mods]
    codecs = S2.codec_registry()
    need = sorted({S6.base_token(t).split("+")[0] for i in ids for t in options_of(items[i])})
    stores = {}
    for f in need:
        enc = S2.load_encoded(qdir / man2["formats"][f]["encoded_file"], lnames)
        if S2.content_sha256(enc) != man2["formats"][f]["content_sha256"]:
            sys.exit(f"{f}: encoded weights differ from the frozen content hash")
        stores[f] = S3.EncodedStore(f, enc)
    glob = {}
    with open(qdir / "quantization_metrics.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["module_name"] == "GLOBAL":
                glob[r["format_key"]] = float(r["sqnr_db"])
    dist_k = {"weights_k8": 8, "stored_k8": 8, "scales_k1": 1}
    summary, trial_rows, t0 = [], [], time.time()
    summary.append({"assignment_id": "FP32", "label": "initial trained model (FP32)", "kind": "reference", "avg_bits_per_weight": 32.0, "protection": "-", "sqnr_clean_db": float("inf"),
                    "sqnr_check_vs_stage2_db": "", "note": "no quantization noise: error 0, SQNR = +inf"})
    for aid in ids:
        opts = options_of(items[aid])
        ms = E2.MixedStore([S6.base_token(t) for t in opts], stores)
        prots = [S6.parse_option(t)[1] for t in opts]
        inj = E2.MixedFaultInjector(model, ms, codecs, "cpu")
        pf = S6.ProtectedFaults(ms, prots)
        meter = Q.SqnrMeter(fp32_w, inj)
        clean = meter.clean_db()
        fmts = sorted(set(S6.parse_option(t)[0] for t in opts))
        check = ""
        if aid.startswith("U_") and len(fmts) == 1 and all(p == "none" for p in prots) and fmts[0] in glob:
            check = clean - glob[fmts[0]]
        dists = [d for d in args.distributions if d != "scales_k1" or E2.has_scale_words([S6.base_token(t) for t in opts])]
        rows = Q.run_assignment(meter, pf, ms, aid, dists, dist_k, args.trials, base_seed, args.draw, recorded)
        trial_rows += rows
        sm = Q.summarize(rows, clean)
        sb = ms.stored_bytes()
        n_w = int(ms.n.sum())
        row = {"assignment_id": aid, "label": ("FINAL assignment" if aid == final["assignment_id"] else aid), "kind": "final" if aid == final["assignment_id"] else "uniform reference",
               "avg_bits_per_weight": 8.0 * sb["total_bytes"] / n_w, "protection": ",".join(sorted(set(prots))), "sqnr_clean_db": clean,
               "sqnr_check_vs_stage2_db": check, "note": ""}
        for d, s in sm.items():
            for k, v in s.items():
                row[f"{d}__{k}"] = v
            same = [r["replicates_recorded_trial"] for r in rows if r["distribution"] == d and r["replicates_recorded_trial"] is not None]
            row[f"{d}__trials_identical_to_recorded"] = f"{sum(same)}/{len(same)}"
        summary.append(row)
        print(f"{aid:14s} clean {clean:7.2f} dB" + "".join(f"  | {d}: median {s['sqnr_median_db']:.2f}, worst {s['sqnr_worst_finite_db']:.2f}, non-finite {s['n_nonfinite']}/{s['n_trials']}" for d, s in sm.items())
              + (f"   [vs Stage 2 GLOBAL {check:+.4f} dB]" if check != "" else ""), flush=True)
    keys = list(OrderedDict((k, None) for r in summary for k in r))
    write_csv(out / "sqnr_summary.csv", [{k: r.get(k, "") for k in keys} for r in summary], keys)
    write_csv(out / "sqnr_trials.csv", trial_rows)
    bad = [r for r in trial_rows if not r["restored_exactly"]]
    mism = [r for r in trial_rows if r["replicates_recorded_trial"] is False]
    dump_json({"checkpoint_sha256": man2["checkpoint"]["sha256"], "stage2_frozen_config_sha256": man2["frozen_config_sha256"], "assignments": ids, "trials_per_distribution": args.trials,
               "base_seed": base_seed, "draw_mode": args.draw, "trials_total": len(trial_rows), "restoration_failures": len(bad), "trials_not_identical_to_recorded": len(mism),
               "definition": Q.__doc__, "seconds": time.time() - t0, "source_commit": git_info()}, out / "sqnr_report.json")
    print(f"{len(trial_rows)} trials, {len(bad)} restoration failures, {len(mism)} trials differing from the recorded end-to-end trials -> {out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
