#!/usr/bin/env python
"""J-FAMP runbook Stage 0 (baseline intake) + Stage 1 (layer inventory) for the supplied FP32 checkpoint.

Inference only (no training, no gradients). Evaluates the checkpoint in FP32 on the 5,000-image validation holdout and on
the official 10,000-image test set, and writes (under results/<run>/jfamp/):

  baseline/        baseline_manifest.json  baseline_metrics.json  confusion_matrix.csv  per_class_metrics.csv
                   checkpoint_validation.txt
  layer_profile/   layer_inventory.csv  layer_inventory.json  layer_storage_summary.csv  parameter_summary.json

  python scripts/stage01_baseline.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch \
      --supplied-by "<name>" --source-location "<where the checkpoint came from>"

Gate 0 / Gate 1 results are printed and stored; the script exits non-zero if a gate fails.
The checkpoint file itself is only read.
"""
import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from fr import data as fdata
from fr.baseline_profile import (checkpoint_checks, evaluate_full, layer_inventory, parameter_summary, reconcile,
                                 sha256_file, sha256_ints, storage_summary)
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_checkpoint, load_model, state_dict_of
from fr.utils import dump_json, env_info, resolve_device

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
RUNBOOK_REFERENCE = {"test_accuracy_percent": 95.11, "test_correct": 9511, "test_total": 10000,
                     "n_quantizable_layers": 21, "n_quantizable_weights": 11164352, "n_parameters_total": 11173962,
                     "fp32_quantizable_bytes": 44657408}


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default=None, help="default results/<run>/jfamp")
    ap.add_argument("--supplied-by", default="not recorded (trained in this project)",
                    help="person who supplied the checkpoint (runbook 2.1)")
    ap.add_argument("--source-location", default="not recorded", help="where the checkpoint came from")
    ap.add_argument("--tolerance-pp", type=float, default=0.05,
                    help="Gate 0: allowed |test accuracy - reference| in percentage points (reference 95.11)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    out = Path(args.out_dir) if args.out_dir else cfg.results_dir() / "jfamp"
    base, prof = out / "baseline", out / "layer_profile"
    base.mkdir(parents=True, exist_ok=True)
    prof.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    ck_path = Path(args.checkpoint)

    # ------------------------------------------------------------------ Stage 0: identity
    ckpt = load_checkpoint(ck_path)
    sd = state_dict_of(ckpt)
    model, _ = load_model(ck_path)                                   # strict=True: raises if keys / shapes differ
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    st = ck_path.stat()
    sha = sha256_file(ck_path)
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    n_train_all = 50000
    tr_idx, va_idx = fdata.split_indices(n_train_all, val_size, split_seed)
    gpu = None
    if device.type == "cuda":
        pr = torch.cuda.get_device_properties(device)
        gpu = {"name": pr.name, "total_memory_bytes": int(pr.total_memory),
               "free_memory_bytes_at_start": int(torch.cuda.mem_get_info(device)[0])}

    # ------------------------------------------------------------------ Stage 0: evaluation (FP32, deterministic)
    test_loader = fdata.get_test_loader(cfg.data_path(), norm, cfg.eval_batch_size, cfg.num_workers, None, False)
    val_loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed,
                                      cfg.num_workers, False, subset=None)
    classes = getattr(test_loader.dataset, "classes", CIFAR10_CLASSES)
    Xt, yt = load_tensors(test_loader, device)
    Xv, yv = load_tensors(val_loader, device)
    with torch.inference_mode():
        test = evaluate_full(model, Xt, yt, len(classes))
        val = evaluate_full(model, Xv, yv, len(classes))
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # ------------------------------------------------------------------ Stage 0: checks
    checks = checkpoint_checks(model, ckpt, sd, RUNBOOK_REFERENCE)
    diff = test["accuracy_percent"] - RUNBOOK_REFERENCE["test_accuracy_percent"]
    gate0 = {"checkpoint_loads_strictly": checks["strict_load"], "output_dim_is_10": checks["output_dim_is_10"],
             "stem_3x3_stride1_no_maxpool": checks["stem_is_3x3_stride1"] and checks["no_maxpool"],
             "parameters_finite": checks["all_parameters_finite"], "logits_finite": test["all_logits_finite"] and val["all_logits_finite"],
             "test_matches_reference_within_tolerance": abs(diff) <= args.tolerance_pp,
             "preprocessing_known": True}
    gate0["PASS"] = all(gate0.values())

    manifest = {
        "checkpoint": {"path": str(ck_path.resolve()), "file_name": ck_path.name, "size_bytes": st.st_size, "sha256": sha,
                       "file_modified_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(timespec="seconds"),
                       "supplied_by": args.supplied_by, "source_location": args.source_location,
                       "top_level_keys": list(ckpt.keys()), "state_dict_tensors": len(sd),
                       "epoch": ckpt.get("epoch"), "best_val_acc_during_training": ckpt.get("best_val_acc"),
                       "best_val_loss_during_training": ckpt.get("best_val_loss"), "git_commit_at_training": ckpt.get("git_commit")},
        "model": {"definition": "fr.model.build_resnet18_cifar", "arch_in_config": (ckpt.get("config") or {}).get("arch"),
                  "constructor": {"num_classes": 10}, "stem": "3x3 conv, stride 1, padding 1, no max-pooling",
                  "output_dim": checks["output_dim"]},
        "training_config": ckpt.get("config"),
        "environment_at_evaluation": env_info(), "gpu": gpu, "evaluation_device": str(device),
        "dataset": {"name": "CIFAR-10", "train_pool": n_train_all, "val_size": val_size, "split_seed": split_seed,
                    "split_rule": "fr.data.split_indices (generator seed 1234 + split_seed)",
                    "val_indices_sha256": sha256_ints(va_idx), "train_indices_sha256": sha256_ints(tr_idx),
                    "test_images": len(yt), "test_set": "official CIFAR-10 test split, never used for selection or tuning"},
        "preprocessing": {"normalization_mean": list(norm[0]) if norm is not None else None,
                          "normalization_std": list(norm[1]) if norm is not None else None,
                          "image_size": "32x32", "channel_order": "RGB", "interpolation": "none (no resize)",
                          "evaluation_transform": repr(fdata.make_transform(norm, train=False)),
                          "training_transform": repr(fdata.make_transform(norm, train=True)),
                          "evaluation_deterministic": True},
        "evaluation": {"precision": "FP32 (no autocast)", "model_eval_mode": True, "gradients_disabled": True,
                       "batch_size": cfg.eval_batch_size, "timestamp_utc": stamp},
        "runbook_reference": RUNBOOK_REFERENCE,
    }
    dump_json(manifest, base / "baseline_manifest.json")

    def split_block(r):
        return {"accuracy_percent": r["accuracy_percent"], "correct": r["correct"], "total": r["total"],
                "cross_entropy": r["cross_entropy"], "per_class": r["per_class"]}
    metrics = {"timestamp_utc": stamp, "checkpoint_sha256": sha, "validation": split_block(val), "test": split_block(test),
               "test_vs_runbook_reference": {"reference_percent": RUNBOOK_REFERENCE["test_accuracy_percent"],
                                             "reference_correct": RUNBOOK_REFERENCE["test_correct"],
                                             "difference_pp": diff, "within_tolerance_pp": args.tolerance_pp},
               "class_names": list(classes)}
    for r in metrics["validation"]["per_class"] + metrics["test"]["per_class"]:
        r["class_name"] = classes[r["class_id"]]
    dump_json(metrics, base / "baseline_metrics.json")
    with open(base / "confusion_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\predicted"] + list(classes))
        for i, row in enumerate(test["confusion"]):
            w.writerow([classes[i]] + [int(v) for v in row])
    write_csv(base / "per_class_metrics.csv",
              [{"class_id": r["class_id"], "class_name": classes[r["class_id"]], "support": r["support"],
                "correct": r["correct"], "accuracy_percent": f"{r['accuracy_percent']:.2f}"}
               for r in test["per_class"]])
    lines = ["Checkpoint validation (J-FAMP runbook Stage 0)", f"file: {ck_path}", f"sha256: {sha}",
             f"evaluated (UTC): {stamp}", "",
             f"strict load of the state dict: {'PASS' if checks['strict_load'] else 'FAIL'}",
             f"  missing keys: {checks['missing_keys']}  unexpected keys: {checks['unexpected_keys']}  "
             f"shape mismatches: {checks['shape_mismatches']}",
             f"output dimension {checks['output_dim']} (must be 10): {'PASS' if checks['output_dim_is_10'] else 'FAIL'}",
             f"stem kernel {checks['stem_kernel']} stride {checks['stem_stride']}, no max-pooling: "
             f"{'PASS' if checks['stem_is_3x3_stride1'] and checks['no_maxpool'] else 'FAIL'}",
             f"NaN/Inf parameters: {checks['nonfinite_parameters'] or 'none'}",
             f"NaN/Inf logits on validation / test: {not val['all_logits_finite']} / {not test['all_logits_finite']}",
             f"model.eval(): {checks['model_in_eval_mode']}; gradients disabled during evaluation (requires_grad False, inference_mode)",
             "", "Comparison with the runbook reference values:"]
    for k, v in checks["comparison_with_runbook_reference"].items():
        lines.append(f"  {k}: expected {v['expected']}, got {v['got']} -> {'match' if v['matches'] else 'DEVIATION'}")
    lines += [f"  test accuracy: reference {RUNBOOK_REFERENCE['test_accuracy_percent']:.2f} %, got "
              f"{test['accuracy_percent']:.2f} % ({test['correct']}/{test['total']}), difference {diff:+.3f} pp "
              f"(tolerance {args.tolerance_pp} pp)",
              f"  validation accuracy (5,000 held-out images): {val['accuracy_percent']:.2f} % ({val['correct']}/{val['total']}); "
              f"best-epoch value recorded during training: {ckpt.get('best_val_acc')}",
              "", "Deviations from the runbook's starting assumption:",
              "  - the checkpoint was trained in this project, not supplied by a teammate (see supplied_by in the manifest)",
              f"  - training ran with deterministic=False; training git state: {ckpt.get('git_commit')}",
              "", f"GATE 0: {'PASS' if gate0['PASS'] else 'FAIL'}  " + json.dumps(gate0)]
    (base / "checkpoint_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ Stage 1
    rows = layer_inventory(model.cpu())
    rec = reconcile(rows, model)
    psum = parameter_summary(model, rows)
    psum["reconciliation"] = rec
    psum["runbook_reference_check"] = {
        "layers": {"expected": 21, "got": len(rows), "matches": len(rows) == 21},
        "quantizable_weights": {"expected": 11164352, "got": psum["quantizable_weights"],
                                "matches": psum["quantizable_weights"] == 11164352},
        "total_parameters": {"expected": 11173962, "got": psum["total_parameters"],
                             "matches": psum["total_parameters"] == 11173962},
        "fp32_quantizable_bytes": {"expected": 44657408, "got": psum["fp32_quantizable_bytes"],
                                   "matches": psum["fp32_quantizable_bytes"] == 44657408}}
    flat = [{**{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}} for r in rows]
    write_csv(prof / "layer_inventory.csv", flat)
    dump_json({"checkpoint_sha256": sha, "ordering": "model definition order", "layers": rows}, prof / "layer_inventory.json")
    write_csv(prof / "layer_storage_summary.csv", storage_summary(rows))
    dump_json(psum, prof / "parameter_summary.json")
    gate1 = all(rec.values()) and all(v["matches"] for v in psum["runbook_reference_check"].values())

    print(f"\nFP32 test        : {test['accuracy_percent']:.2f} %  ({test['correct']}/{test['total']})  CE {test['cross_entropy']:.4f}")
    print(f"FP32 validation  : {val['accuracy_percent']:.2f} %  ({val['correct']}/{val['total']})  CE {val['cross_entropy']:.4f}")
    print(f"reference 95.11 %: difference {diff:+.3f} pp")
    print(f"layers {len(rows)}, quantizable weights {psum['quantizable_weights']}, total params {psum['total_parameters']}")
    print(f"GATE 0 (baseline): {'PASS' if gate0['PASS'] else 'FAIL'}   GATE 1 (inventory): {'PASS' if gate1 else 'FAIL'}")
    print("written:", out)
    sys.exit(0 if (gate0["PASS"] and gate1) else 1)


if __name__ == "__main__":
    main()
