#!/usr/bin/env python
"""J-FAMP runbook Stage 4: global and layer-wise fault screening (Sections 6.1 - 6.3). Validation images only.

Runs (resumably) every format x scope x fault group x k configuration with the same trial budget and sampling policy, on the
frozen Stage 2 encoded weights, using the Stage 3 injector (bit flips in the stored encoded bytes). Refuses to start unless the
Stage 3 gate passed for the chosen formats on the same checkpoint / frozen configuration. The test set is not touched.

  python scripts/stage4_fault_screening.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
  python scripts/stage4_fault_screening.py ... --summarize-only      # regenerate every CSV from fault_trials.jsonl

Outputs under results/<run>/jfamp/fault_screening/:
  fault_trials.jsonl          one record per trial (all Section 5.3 fields, seed, flips)
  fault_summary.csv           layer-wise: n, mean, median, std, p90, worst, failure counts per layer x format x group x k
  global_fault_summary.csv    the same metrics for the global (all-layer) scope
  layer_fault_heatmap.csv     layer x format/group/k matrix of mean accuracy drop and p90
  fault_campaign_manifest.json  run settings, completion counts, skipped configurations, software versions
  stage4_report.md            short auto-generated overview
Re-running the same command resumes: finished trial ids are skipped; completeness is verified at the end.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import math
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from fr import data as fdata
from fr import stage2 as S2
from fr import stage3 as S3
from fr import stage4 as S4
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, env_info, git_info, resolve_device

PRIMARY = ["e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt"]


def cell(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else "inf" if v == math.inf else "-inf" if v == -math.inf else repr(round(v, 8))
    return "" if v is None else v


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k, "")) for k in fields})


def write_outputs(out, records, expected, layer_meta, manifest_extra, spe):
    rows = S4.summarize(records)
    glob, lay = S4.split_global(rows)
    write_csv(out / "global_fault_summary.csv", glob, S4.SUMMARY_FIELDS)
    write_csv(out / "fault_summary.csv", lay, S4.SUMMARY_FIELDS)
    header, hm = S4.heatmap(rows, layer_meta)
    write_csv(out / "layer_fault_heatmap.csv", hm, header)
    comp = S4.verify_completeness(records, expected)
    return rows, glob, lay, comp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None, help="default results/<run>/jfamp/quantization")
    ap.add_argument("--stage3-dir", default=None, help="default results/<run>/jfamp/fault_injection")
    ap.add_argument("--out-dir", default=None, help="default results/<run>/jfamp/fault_screening")
    ap.add_argument("--inventory", default=None, help="default results/<run>/jfamp/layer_profile/layer_inventory.csv")
    ap.add_argument("--n-images", type=int, default=1000, help="validation images per trial (same set as Stage 3)")
    ap.add_argument("--trials", type=int, default=20, help="independent trials per configuration (runbook: 20-30)")
    ap.add_argument("--formats", nargs="+", default=PRIMARY, choices=list(S3.SPECS))
    ap.add_argument("--include-comparison", action="store_true", help="also screen E4M11-FTZ (comparison format)")
    ap.add_argument("--no-scale-scope", action="store_true", help="skip the INT scale-word extension scope")
    ap.add_argument("--summarize-only", action="store_true", help="regenerate summaries from fault_trials.jsonl, run no trials")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    s3dir = Path(args.stage3_dir) if args.stage3_dir else base / "fault_injection"
    out = Path(args.out_dir) if args.out_dir else base / "fault_screening"
    out.mkdir(parents=True, exist_ok=True)
    inv_path = Path(args.inventory) if args.inventory else base / "layer_profile" / "layer_inventory.csv"
    ck = Path(args.checkpoint)
    formats = list(args.formats) + (["e4m11_ftz"] if args.include_comparison and "e4m11_ftz" not in args.formats else [])

    # ------------------------------------------------------------------------------------------ inputs and gates
    man2 = json.load(open(qdir / "quantizer_manifest.json"))
    ck_sha = sha256_file(ck)
    if man2["checkpoint"]["sha256"] != ck_sha:
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    frozen = hashlib.sha256(json.dumps({"checkpoint_sha256": ck_sha, "formats": {k: [v["config"], v["content_sha256"]] for k, v in man2["formats"].items()}},
                                       sort_keys=True).encode()).hexdigest()
    if frozen != man2["frozen_config_sha256"]:
        sys.exit("Stage 2 manifest does not reproduce its frozen-configuration hash")
    val3 = json.load(open(s3dir / "codec_fault_validation.json"))
    if val3["checkpoint_sha256"] != ck_sha or val3["stage2_frozen_config_sha256"] != frozen:
        sys.exit("Stage 3 validation was run on a different checkpoint / frozen configuration")
    not_ok = [f for f in formats if val3["gate3_per_format"].get(f) != "PASS"]
    if not_ok:
        sys.exit(f"Gate 3 has not passed for {not_ok}; Stage 4 must not start (runbook Gate 3)")

    model, ckpt = load_model(ck)
    model.eval()
    mods = quantizable_modules(model)
    layer_names = [f"{n}.weight" for n, _ in mods]
    module_names = [n for n, _ in mods]
    inventory = list(csv.DictReader(open(inv_path))) if inv_path.exists() else None
    layer_meta = inventory or [{"layer_id": i, "module_name": n, "num_weights": int(m.weight.numel())} for i, (n, m) in enumerate(mods)]
    if [r["module_name"] for r in layer_meta] != module_names:
        sys.exit("layer order differs from the Stage 1 inventory")

    # ------------------------------------------------------------------------------------------ configurations
    all_cfgs, skipped = {}, []
    for f in formats:
        cfgs, sk = S4.enumerate_configs(f, module_names, include_scales=not args.no_scale_scope)
        all_cfgs[f] = cfgs
        skipped += sk
    expected = [i for f in formats for i in S4.expected_ids(all_cfgs[f], args.trials)]
    n_cfg = sum(len(v) for v in all_cfgs.values())
    print(f"{len(formats)} formats, {n_cfg} configurations x {args.trials} trials = {len(expected)} trials "
          f"({', '.join(f'{f}:{len(all_cfgs[f])}' for f in formats)} configurations)")

    log = S4.TrialLog(out / "fault_trials.jsonl")
    if log.duplicates:
        sys.exit(f"fault_trials.jsonl contains duplicate trial ids: {log.duplicates[:5]}")
    manifest_path = out / "fault_campaign_manifest.json"
    prev = json.load(open(manifest_path)) if manifest_path.exists() else {}
    run_id = prev.get("run_id") or (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    experiment_id = "jfamp-resnet18-cifar10-stage4-fault-screening"
    sessions = list(prev.get("sessions", []))
    clean_info = dict(prev.get("clean_per_format", {}))
    t_start = time.time()
    session = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "new_trials": 0,
               "skipped_already_done": 0, "git": git_info(), "summarize_only": args.summarize_only}

    if not args.summarize_only:
        norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
        val_size, split_seed = split_params(ckpt, cfg)
        loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False,
                                      subset=args.n_images)
        X, y = load_tensors(loader, device)
        print(f"run {run_id}: {int(y.numel())} validation images, device {device}")
        for fmt in formats:
            codec = S2.make_codecs(True)[fmt]
            stored = S2.load_encoded(qdir / man2["formats"][fmt]["encoded_file"], layer_names)
            if S2.content_sha256(stored) != man2["formats"][fmt]["content_sha256"]:
                sys.exit(f"{fmt}: encoded weights differ from the frozen content hash")
            inj = S3.EncodedFaultInjector(model, S3.EncodedStore(fmt, stored), codec, device)
            clean = S3.evaluate(inj.model, X, y)
            clean_info[fmt] = {"clean_accuracy": clean["accuracy"], "clean_loss": clean["loss"], "sample_count": int(y.numel())}
            ctx = {"experiment_id": experiment_id, "run_id": run_id,
                   "model": "resnet18_cifar (CIFAR-style: 3x3 stem, no max-pool, 10 classes)",
                   "dataset": f"CIFAR-10 validation holdout, first {int(y.numel())} images", "checkpoint_hash": ck_sha,
                   "codec_version": f"{fmt}|stage2-frozen:{frozen[:12]}|{S3.INJECTOR_VERSION}"}
            # ---- 6.1 first step: model-level smoke test per format and fault group (not recorded in fault_trials.jsonl)
            Lall = list(range(len(layer_names)))
            for g in [g for g in S3.format_groups(fmt) if S3.group_bits(fmt, g)]:
                r = S3.run_trial(inj, X, y, clean, ctx, fmt=fmt, scope_layers=Lall, scope_name="GLOBAL", scope_kind="weights", group=g,
                                 k=8, trial=1_000_000, base_seed=cfg.seed)
                if not r["restored_exactly"] or not r["flip_applied_to_encoded_bytes"]:
                    sys.exit(f"pre-flight smoke test failed for {fmt}/{g}")
            print(f"\n=== {fmt}: clean {clean['accuracy']:.2f} % / loss {clean['loss']:.4f} on {int(y.numel())} images; "
                  f"pre-flight smoke passed; {len(all_cfgs[fmt])} configurations")
            t0 = time.time()
            res = S4.run_configs(inj, X, y, clean, ctx, all_cfgs[fmt], args.trials, cfg.seed, log)
            session["new_trials"] += res["new"]
            session["skipped_already_done"] += res["skipped_already_done"]
            clean_info[fmt]["seconds_this_session"] = time.time() - t0
            print(f"    {res['new']} new trials, {res['skipped_already_done']} already done, {time.time() - t0:.0f} s")
            del inj
            if device.type == "cuda":
                torch.cuda.empty_cache()
    log.close()
    session["seconds"] = time.time() - t_start
    session["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    sessions.append(session)

    # ------------------------------------------------------------------------------------------ summaries from raw records
    records = S4.load_records(out / "fault_trials.jsonl")
    rows, glob, lay, comp = write_outputs(out, records, expected, layer_meta, {}, None)
    total_new = sum(s["new_trials"] for s in sessions)
    sec_total = sum(s["seconds"] for s in sessions if not s["summarize_only"])
    spe = sec_total / total_new if total_new else None
    by_fmt = {f: {"configurations": len(all_cfgs[f]), "global": sum(c["level"] == "global" for c in all_cfgs[f]),
                  "layerwise": sum(c["level"] == "layerwise" for c in all_cfgs[f]),
                  "scale_scope": sum(c["scope_kind"] == "scales" for c in all_cfgs[f]),
                  "expected_trials": len(all_cfgs[f]) * args.trials,
                  "recorded_trials": sum(1 for r in records if r["format"] == f),
                  **clean_info.get(f, {})} for f in formats}
    flag_totals = {n: sum(n in r["output_flag"].split(",") for r in records) for n in S4.FLAG_NAMES}
    manifest = {
        "stage": "J-FAMP runbook Stage 4 - global and layer-wise fault screening (Sections 6.1-6.3)",
        "label": "SCREENING on validation images (not paper-grade coverage); the test set is not used",
        "run_id": run_id, "experiment_id": experiment_id, "checkpoint_sha256": ck_sha, "stage2_frozen_config_sha256": frozen,
        "stage3": {"run_id": val3["run_id"], "gate3_per_format": val3["gate3_per_format"], "injector_version": val3["injector_version"]},
        "settings": {"trials_per_configuration": args.trials, "k_values": list(S4.K_VALUES), "validation_images": args.n_images,
                     "base_seed": cfg.seed, "formats": formats, "scale_scope_extension": not args.no_scale_scope,
                     "sampling": "k distinct elements without replacement within the scope, bit uniform within the group; one derived "
                                 "seed per trial (see fault_protocol.json of Stage 3)",
                     "non_finite_policy": "images with non-finite logits count as incorrect; fault_loss is null if any logit is "
                                          "non-finite; such trials are kept and counted as failures",
                     "groups": {f: [g for g in S3.format_groups(f) if S3.group_bits(f, g)] for f in formats},
                     "preflight": "one k=8 global trial per format and fault group before the campaign (not recorded in the trial file)"},
        "metric_definitions": {
            "drop_pp": "clean accuracy - fault accuracy, percentage points, same 1,000 images and same model state",
            "rel_drop_percent": "100 * drop_pp / clean accuracy (reported only alongside pp)",
            "loss_increase": "fault loss - clean loss, over trials whose fault loss is finite (n_loss_finite)",
            "failure": "trial with non-finite weights, activations, logits or loss; failure_rate = failures / n_trials; the four "
                       "causes are counted separately",
            "std": "sample standard deviation (ddof = 1)", "p90": "90th percentile (linear interpolation) of the accuracy drop",
            "worst": "maximum accuracy drop", "sem": "std / sqrt(n_trials)"},
        "fairness_notes": ["clean accuracy differs between formats on the same images (see clean_per_format); compare formats by "
                           "fault accuracy and by drop together with the clean accuracy, not by drop alone",
                           "a 1,000-image validation screen is not comparable with a 10,000-image test result"],
        "clean_per_format": clean_info, "per_format": by_fmt,
        "configurations": {"total": n_cfg, "trials_total_expected": len(expected)},
        "skipped_configurations": skipped + [{"format": "e4m11_ftz", "reason": "comparison format, not screened unless --include-comparison"}
                                              for _ in [0] if "e4m11_ftz" not in formats],
        "completion": comp, "failure_counts_by_cause": flag_totals,
        "trials_recorded": len(records), "mean_seconds_per_trial": spe, "sessions": sessions,
        "environment": env_info(), "git": git_info(), "injector_version": S3.INJECTOR_VERSION,
        "robustness_score": "not defined (no composite score is computed; underlying metrics are preserved)"}
    dump_json(manifest, manifest_path)

    # ------------------------------------------------------------------------------------------ short report
    L = ["# Stage 4 - fault screening (validation images, screening protocol)", "",
         f"run `{run_id}`; {len(records)} of {len(expected)} expected trials recorded; completeness: "
         f"**{'COMPLETE' if comp['complete'] else 'INCOMPLETE'}** (duplicates {len(comp['duplicates'])}, missing {comp['n_missing']})", "",
         "## Global scope (all weights), mean accuracy drop in pp (p90) over " + str(args.trials) + " trials", "",
         "| format | group | k | clean % | mean drop | p90 | worst | failures |", "|---|---|---|---|---|---|---|---|"]
    for r in glob:
        if r["scope_kind"] == "weights":
            L.append(f"| {r['format']} | {r['fault_group']} | {r['k']} | {r['clean_accuracy']:.2f} | {r['drop_pp_mean']:.2f} | "
                     f"{r['drop_pp_p90']:.2f} | {r['drop_pp_worst']:.2f} | {r['failures']}/{r['n_trials']} |")
    L += ["", "Full tables: `global_fault_summary.csv`, `fault_summary.csv`, `layer_fault_heatmap.csv`; raw records: `fault_trials.jsonl`."]
    (out / "stage4_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\ncompleteness: {'COMPLETE' if comp['complete'] else 'INCOMPLETE'} ({comp['recorded']}/{comp['expected']} trials; "
          f"duplicates {len(comp['duplicates'])}, missing {comp['n_missing']})")
    print("written:", out)
    sys.exit(0 if comp["complete"] else 2)


if __name__ == "__main__":
    main()
