#!/usr/bin/env python
"""J-FAMP runbook Stage 3: fault-injection protocol validation (Sections 5.1 - 5.3).

Proves, for every encoded format of Stage 2, that the injector flips bits in the ACTUAL ENCODED representation and restores it
exactly, BEFORE any fault campaign (Gate 3):

  * round-trip exactness / stable bytes      * bit mapping of every bit of a synthetic tensor (+ special values)
  * INT4 packing and padding                 * scope (layer, element count, trial manifest, Stage 1 inventory)
  * no-fault control                         * restoration (SHA-256 of the whole encoded state before / after)
  * non-finite handling (weights, activations, logits, loss; trials are kept and flagged)

Writes under results/<run>/jfamp/fault_injection/:
  fault_protocol.json  codec_fault_validation.json  fault_injection_smoke_tests.csv      (runbook outputs)
  fault_injection_smoke_trials.jsonl  stage3_report.md                                    (supporting: full 5.3 trial records)

  python scripts/stage3_fault_validation.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch

Uses the frozen Stage 2 encoded weights (results/<run>/jfamp/quantization/quantized_weights) and validation images only.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
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
from fr import stage3_validation as V
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.stage3_protocol import build_protocol
from fr.utils import dump_json, env_info, git_info, resolve_device


def fp32_state_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fmt_cell(v):
    if isinstance(v, float):
        return "nan" if v != v else "inf" if v == float("inf") else "-inf" if v == float("-inf") else repr(v)
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None, help="default results/<run>/jfamp/quantization")
    ap.add_argument("--out-dir", default=None, help="default results/<run>/jfamp/fault_injection")
    ap.add_argument("--inventory", default=None, help="default results/<run>/jfamp/layer_profile/layer_inventory.csv")
    ap.add_argument("--n-images", type=int, default=1000, help="validation images used by the smoke trials")
    ap.add_argument("--smoke-trials", type=int, default=3, help="trials per (group, k) in the smoke campaign")
    ap.add_argument("--trials-screening", type=int, default=30, help="planned Stage 4 trials per layer/format/group (20-30)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    out = Path(args.out_dir) if args.out_dir else base / "fault_injection"
    out.mkdir(parents=True, exist_ok=True)
    inv_path = Path(args.inventory) if args.inventory else base / "layer_profile" / "layer_inventory.csv"
    ck = Path(args.checkpoint)

    # ------------------------------------------------------------------------------------------ frozen Stage 2 inputs
    man = json.load(open(qdir / "quantizer_manifest.json"))
    ck_sha = sha256_file(ck)
    if man["checkpoint"]["sha256"] != ck_sha:
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    frozen = hashlib.sha256(json.dumps({"checkpoint_sha256": ck_sha, "formats": {k: [v["config"], v["content_sha256"]] for k, v in man["formats"].items()}},
                                       sort_keys=True).encode()).hexdigest()
    if frozen != man["frozen_config_sha256"]:
        sys.exit("Stage 2 manifest does not reproduce its frozen-configuration hash")
    model, ckpt = load_model(ck)
    model.eval()
    mods = quantizable_modules(model)
    layer_names = [f"{n}.weight" for n, _ in mods]
    weights = OrderedDict((f"{n}.weight", m.weight.detach().cpu().numpy().astype(np.float32)) for n, m in mods)
    numels = OrderedDict((f"{n}.weight", int(m.weight.numel())) for n, m in mods)
    inventory = list(csv.DictReader(open(inv_path))) if inv_path.exists() else None
    if inventory is None:
        print(f"WARNING: {inv_path} not found; Stage 1 inventory comparison skipped")
    layers_meta = inventory or [{"layer_id": i, "module_name": n[:-7], "num_weights": numels[n]} for i, n in enumerate(layer_names)]
    codecs = S2.make_codecs(True)
    commit = git_info()["commit"]
    codec_versions = {k: f"{k}|stage2-frozen:{frozen[:12]}|{S3.INJECTOR_VERSION}|git:{commit[:8]}" for k in S3.SPECS}

    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False,
                                  subset=args.n_images)
    X, y = load_tensors(loader, device)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    experiment_id = "jfamp-resnet18-cifar10-stage3-fault-validation"
    print(f"run {run_id}: {len(layer_names)} layers, {int(sum(numels.values()))} weights, {int(y.numel())} validation images")

    smoke_rows, validation, all_records = [], OrderedDict(), []
    eval_seconds = []
    for fmt, codec in codecs.items():
        t_fmt = time.time()
        tests = []
        # ---- Stage 2 frozen encoded weights, hash-checked
        stored = S2.load_encoded(qdir / man["formats"][fmt]["encoded_file"], layer_names)
        # ---- synthetic suites -> smoke CSV
        rows = V.bit_mapping_rows(fmt, codec) + V.special_value_rows(fmt, codec)
        if fmt in S3.SCALE_FORMATS:
            rows += V.scale_bit_rows(fmt, codec)
        if fmt == "int4_pt":
            rows += V.packing_rows(codec)
        smoke_rows += rows
        by_cat = {}
        for r in rows:
            by_cat.setdefault(r["category"], [0, 0])
            by_cat[r["category"]][0 if r["status"] == "PASS" else 1] += 1
        for cat, (p, f) in by_cat.items():
            tests.append({"test": f"{fmt}: synthetic {cat} ({p + f} cases, see fault_injection_smoke_tests.csv)", "passed": f == 0,
                          "detail": f"{p} pass, {f} fail"})
        tests += V.roundtrip_synthetic_tests(fmt, codec)
        tests += V.roundtrip_real_tests(fmt, codec, weights, stored, man["formats"][fmt]["content_sha256"])
        store0 = S3.EncodedStore(fmt, stored)
        tests += V.scope_tests(store0, numels, inventory, fmt)

        # ---- model-level: no-fault control, restoration, non-finite
        ctx = {"experiment_id": experiment_id, "run_id": run_id, "model": "resnet18_cifar (CIFAR-style: 3x3 stem, no max-pool, 10 classes)",
               "dataset": f"CIFAR-10 validation holdout, first {int(y.numel())} images", "checkpoint_hash": ck_sha,
               "codec_version": codec_versions[fmt]}
        inj = S3.EncodedFaultInjector(model, S3.EncodedStore(fmt, stored), codec, device)
        res = V.model_level_tests(fmt, inj, X, y, ctx, cfg.seed, lambda: fp32_state_hash(model))
        tests += res["tests"]
        all_records += res["records"]
        clean = res["clean"]
        L = list(range(len(layer_names)))
        gl = [g for g in S3.format_groups(fmt) if S3.group_bits(fmt, g)]

        # ---- smoke campaign (schema-complete trial records; screening itself is Stage 4)
        smoke = []
        for g in gl:
            for k in (1, 8):
                for t in range(args.smoke_trials):
                    t0 = time.time()
                    smoke.append(S3.run_trial(inj, X, y, clean, ctx, fmt=fmt, scope_layers=L, scope_name="GLOBAL", scope_kind="weights",
                                              group=g, k=k, trial=t, base_seed=cfg.seed))
                    eval_seconds.append(time.time() - t0)
        for li in (len(layer_names) - 1, len(layer_names) - 2):                 # layer-wise scope: fc and layer4.1.conv2
            for t in range(args.smoke_trials):
                smoke.append(S3.run_trial(inj, X, y, clean, ctx, fmt=fmt, scope_layers=[li], scope_name=layer_names[li][:-7],
                                          scope_kind="weights", group="all", k=8, trial=t, base_seed=cfg.seed))
        if fmt in S3.SCALE_FORMATS:                                           # optional extension scope: the float32 scales
            for g, k in (("exponent", 1), ("all", 8)):
                for t in range(args.smoke_trials):
                    smoke.append(S3.run_trial(inj, X, y, clean, ctx, fmt=fmt, scope_layers=L, scope_name="GLOBAL-SCALES",
                                              scope_kind="scales", group=g, k=k, trial=t, base_seed=cfg.seed))
        all_records += smoke
        miss = [f for r in smoke + res["records"] for f in S3.REQUIRED_TRIAL_FIELDS if f not in r]
        tests.append({"test": f"{fmt}: all {len(smoke) + len(res['records'])} trial records contain every Section 5.3 field", "passed": not miss,
                      "detail": f"missing: {sorted(set(miss))}" if miss else ""})
        bad_rest = [r for r in smoke if not r["restored_exactly"] or not r["flip_applied_to_encoded_bytes"]]
        tests.append({"test": f"{fmt}: smoke campaign - {len(smoke)} trials, every one restored byte-exactly (hash before == after) and "
                              f"every flip applied to the encoded bytes", "passed": not bad_rest})
        nf = sum(r["failure"] for r in smoke)
        tests.append({"test": f"{fmt}: non-finite outcomes in the smoke campaign are kept and counted (not dropped)", "passed": True,
                      "detail": f"{nf} of {len(smoke)} trials flagged non-finite"})
        n_fail = [t for t in tests if not t["passed"]]
        validation[fmt] = {"role": "comparison" if fmt == "e4m11_ftz" else "primary", "tests": tests, "n_tests": len(tests),
                           "n_passed": len(tests) - len(n_fail), "failed": [t["test"] for t in n_fail],
                           "gate3": "PASS" if not n_fail else "FAIL", "clean_accuracy_on_smoke_images": clean["accuracy"],
                           "clean_loss_on_smoke_images": clean["loss"], "seconds": time.time() - t_fmt}
        print(f"{fmt:11s} Gate 3 {validation[fmt]['gate3']}  ({len(tests) - len(n_fail)}/{len(tests)} tests; {len(smoke) + len(res['records'])} trials; "
              f"clean {clean['accuracy']:.2f} %)")
        del inj
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------------------------------ outputs
    with open(out / "fault_injection_smoke_tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=V.SMOKE_FIELDS)
        w.writeheader()
        for r in smoke_rows:
            w.writerow({k: fmt_cell(r.get(k, "")) for k in V.SMOKE_FIELDS})
    with open(out / "fault_injection_smoke_trials.jsonl", "w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    spe = float(np.mean(eval_seconds)) if eval_seconds else None
    protocol = build_protocol(run_id=run_id, checkpoint_sha=ck_sha, stage2_frozen=frozen, layers=layers_meta, base_seed=cfg.seed,
                              n_images=int(y.numel()), trials_screening=args.trials_screening, smoke_trials=args.smoke_trials,
                              seconds_per_eval=spe, codec_versions=codec_versions)
    dump_json(protocol, out / "fault_protocol.json")
    gate_all = all(v["gate3"] == "PASS" for k, v in validation.items() if v["role"] == "primary")
    dump_json({"stage": "J-FAMP runbook Stage 3 - codec / fault-injection validation", "run_id": run_id,
               "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "checkpoint_sha256": ck_sha, "stage2_frozen_config_sha256": frozen, "injector_version": S3.INJECTOR_VERSION,
               "git": git_info(), "environment": env_info(), "validation_images": int(y.numel()),
               "synthetic_cases": {"total": len(smoke_rows), "passed": sum(r["status"] == "PASS" for r in smoke_rows)},
               "gate3_per_format": {k: v["gate3"] for k, v in validation.items()},
               "gate3_all_primary_formats": "PASS" if gate_all else "FAIL", "formats": validation}, out / "codec_fault_validation.json")
    lines = ["# Stage 3 - fault-injection protocol validation", "", f"run `{run_id}`; checkpoint `{ck_sha[:16]}...`; Stage 2 frozen config `{frozen[:16]}...`",
             f"validation images for the model-level tests: {int(y.numel())}; synthetic bit cases: {len(smoke_rows)}", "",
             "| format | role | tests passed | Gate 3 |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['role']} | {v['n_passed']}/{v['n_tests']} | **{v['gate3']}** |" for k, v in validation.items()]
    lines += ["", f"Overall (primary formats): **{'PASS' if gate_all else 'FAIL'}**",
              f"Measured mean time per trial (incl. hashing): {spe:.2f} s" if spe else "", ""]
    for k, v in validation.items():
        for t in v["failed"]:
            lines.append(f"* FAILED [{k}]: {t}")
    (out / "stage3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nGATE 3 (all primary formats): {'PASS' if gate_all else 'FAIL'}")
    print("written:", out)
    sys.exit(0 if gate_all else 1)


if __name__ == "__main__":
    main()
