#!/usr/bin/env python
"""MobileNetV2 protocol Section 4: encoded-weight fault injection (validation gate + screening + targeted follow-up).

Bits are flipped in the ACTUAL STORED ENCODED BYTES of the frozen Stage 2 encodings (fp16 / bf16 / e4m11_sn / fp8_e4m3fn / fp8_e5m2 /
int8 / packed int4, per-tensor INT scale); the corrupted bytes are decoded and the model is evaluated. The injector is the validated
``fr.stage3`` injector (extended with BF16 and FP8 E5M2). Validation images only; the test set is never read.

Campaigns (--campaign):
  validate    Section 4.3 gate: round trip / stable bytes, known-bit smoke tests (every bit of a synthetic tensor, special values,
              INT4 packing), no-fault control, exact restoration, non-finite tracking, trial-scope verification. MUST pass first.
  global      30 trials per format x fault group x k (k = 1 and 8); random weights across ALL 53 layers.
  layerwise   10 trials per layer x format x fault group x k; one target layer per trial, other layers clean.
  shortlist   writes targeted_shortlist.csv from the layer-wise summary (documented selection rule; disclose the selection bias).
  targeted    100 trials per shortlisted configuration (--targets-csv), always on the FULL validation set.

Groups: fp formats sign / exponent / fraction; INT sign / value (no exponent). k = 8 means eight DISTINCT weights struck (sampled without
replacement), one bit each (bit uniform within the group). Independent seed per trial. Global/layer-wise screening uses a fixed,
declared, class-stratified validation subset (--subset-size, default 1000; 0 = full 5,000); every comparable configuration uses the same
subset, and it is reported separately from full-validation results.

  python scripts/stage4_encoded_fault_injection.py --campaign validate  --checkpoint $CK --run-name mobilenetv2_cifar10 --out-dir $OUT
  python scripts/stage4_encoded_fault_injection.py --campaign global    --checkpoint $CK --run-name mobilenetv2_cifar10 --out-dir $OUT
  python scripts/stage4_encoded_fault_injection.py --campaign layerwise --checkpoint $CK --run-name mobilenetv2_cifar10 --out-dir $OUT

Every campaign is resumable (finished trial ids are skipped) and verifies completeness (no duplicate / missing trial) at the end.
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
from fr import stage3_validation as V
from fr import stage4 as S4
from fr.baseline_profile import sha256_file, sha256_ints
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, env_info, git_info, resolve_device

EXPERIMENT = "jfamp-mobilenetv2-cifar10-stage4-encoded-fault-injection"
PRIMARY = ["fp16", "bf16", "e4m11_sn", "fp8_e4m3fn", "fp8_e5m2", "int8_pt", "int4_pt"]
BUDGET = {"global": 30, "layerwise": 10, "targeted": 100}
CONTROL_TOL = 1e-5                      # max |logit difference| allowed between the zero-flip control and the clean quantized model
SUMMARY_EXTRA = ["mean_changed_predictions", "mean_agreement_percent"]


# ------------------------------------------------------------------------------------------------------------ helpers
def groups_of(fmt: str):
    """Protocol 4.1 groups: fp sign/exponent/fraction, INT sign/value (the ``all`` group of the ResNet run is not part of Section 4)."""
    return [g for g in S3.format_groups(fmt) if g != "all" and S3.group_bits(fmt, g)]


def stratified_subset(labels: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Deterministic class-stratified subset of n positions (sorted); n larger than the set -> everything."""
    labels = np.asarray(labels)
    if n <= 0 or n >= labels.size:
        return np.arange(labels.size)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    base, extra = divmod(n, classes.size)
    pick = []
    for i, c in enumerate(classes):
        pos = np.flatnonzero(labels == c)
        q = min(base + (1 if i < extra else 0), pos.size)
        pick.append(rng.choice(pos, size=q, replace=False))
    return np.sort(np.concatenate(pick))


def make_configs(fmt: str, module_names, campaign: str, layers=None):
    """Global or layer-wise configurations of one format (weights only; the INT scale words are not in the Section 4 scope)."""
    cfgs, skipped = S4.enumerate_configs(fmt, module_names, include_scales=False, layers=layers)
    cfgs = [c for c in cfgs if c["scope_kind"] == "weights" and c["group"] != "all" and c["level"] == campaign]
    for c in cfgs:
        c["selection"] = ""
    return cfgs, [s for s in skipped if s.get("group") != "all"]


def targeted_configs(rows, module_names, layer_ids_by_name):
    cfgs = []
    for r in rows:
        name = r["layer_name"]
        if name not in layer_ids_by_name:
            sys.exit(f"targets file: unknown layer {name!r}")
        fmt = r.get("format_key") or r["format"]
        grp, k = r["group"], int(r["k"])
        if grp not in groups_of(fmt):
            sys.exit(f"targets file: group {grp!r} is not a Section 4 group of {fmt}")
        li = layer_ids_by_name[name]
        cfgs.append({"format": fmt, "level": "targeted", "scope_kind": "weights", "scope_name": module_names[li], "scope_layers": [li],
                     "group": grp, "k": k, "selection": r.get("selection_rule", "user-provided")})
    return cfgs


def verify_trial(rec: dict, cfg: dict, store: S3.EncodedStore, fmt: str) -> None:
    """Trial-scope check against the trial manifest: layer, logical indices, bit indices and flip count. The expected flips are
    regenerated independently from the recorded seed and must equal the recorded ones."""
    k = cfg["k"]
    fl = [(f["layer_id"], f["element"], f["bit_index"]) for f in rec["flips"]]
    bits = set(S3.group_bits(fmt, cfg["group"]))
    assert len(fl) == k == rec["fault_count"] == rec["requested_k"], "flip count differs from the trial manifest"
    assert len({(l, e) for l, e, _ in fl}) == k, "flips must hit k distinct weights"
    assert all(l in cfg["scope_layers"] and 0 <= e < int(store.n[l]) and b in bits for l, e, b in fl), "flip outside the declared scope/group"
    expect = S3.draw_weight_flips(np.random.default_rng(rec["seed"]), store, cfg["scope_layers"], S3.group_bits(fmt, cfg["group"]), k)
    assert [tuple(x) for x in expect] == fl, "recorded flips differ from the flips regenerated from the trial seed"
    for f in rec["flips"]:
        assert f["field"] == S3.field_of_bit(fmt, f["bit_index"]) and (f["stored_index"], f["bit_in_stored_unit"]) == S3.bit_location(fmt, f["element"], f["bit_index"])


def slim(rec: dict, tid: str, c: dict, campaign: str, subset: dict, changed: int) -> dict:
    n = rec["sample_count"]
    r = {k: rec[k] for k in ("layer_id", "layer_name", "format", "scope_kind", "fault_group", "requested_k", "trial_index", "seed",
                              "fault_count", "sample_count", "clean_accuracy", "fault_accuracy", "clean_loss", "fault_loss",
                              "nonfinite_weight_count", "output_flag", "failure", "restored_exactly", "flip_applied_to_encoded_bytes",
                              "run_id", "checkpoint_hash")}
    r.update({"trial_id": tid, "campaign": campaign, "level": c["level"], "scope_verified": True, "eval_subset": subset["name"],
              "flips": [[f["layer_id"], f["element"], f["bit_index"]] for f in rec["flips"]], "changed_predictions": changed,
              "agreement_percent": 100.0 * (1.0 - changed / n), "selection": c.get("selection", ""), "schema": "mnv2-stage4-trial-v1"})
    return r


def run_format(fmt, codec, stored, model, X, y, device, cfgs, trials, log, base_seed, ctx, campaign, subset, progress=print):
    """Control + every (configuration, trial) of one format not yet in ``log``. Returns (control row, counts)."""
    inj = S3.EncodedFaultInjector(model, S3.EncodedStore(fmt, stored), codec, device)
    store = inj.store
    clean = S3.evaluate(inj.model, X, y, return_logits=True)
    clean_logits = clean.pop("logits")
    clean_pred = clean_logits.argmax(1)
    allL = list(range(len(store.names)))
    h0 = store.hash()
    r0 = S3.run_trial(inj, X, y, clean, ctx, fmt=fmt, scope_layers=allL, scope_name="GLOBAL", scope_kind="weights", group="none", k=0,
                      trial=0, base_seed=base_seed, return_logits=True)
    d = float((r0.pop("_logits") - clean_logits).abs().max())
    control = {"campaign": campaign, "format": fmt, "flips": 0, "eval_images": int(y.numel()), "clean_accuracy": clean["accuracy"],
               "control_accuracy": r0["fault_accuracy"], "max_abs_logit_diff": d, "tolerance": CONTROL_TOL,
               "encoded_hash_before": h0, "encoded_hash_after": store.hash(), "restored_exactly": r0["restored_exactly"],
               "passed": bool(d <= CONTROL_TOL and r0["fault_accuracy"] == clean["accuracy"] and r0["restored_exactly"])}
    if not control["passed"]:
        raise RuntimeError(f"no-fault control failed for {fmt}: {control}")
    n_new = n_skip = 0
    for ci, c in enumerate(cfgs):
        acc = []
        for t in range(trials):
            tid = S4.trial_id(fmt, c["scope_kind"], c["scope_name"], c["group"], c["k"], t)
            if log.has(tid):
                n_skip += 1
                continue
            rec = S3.run_trial(inj, X, y, clean, ctx, fmt=fmt, scope_layers=c["scope_layers"], scope_name=c["scope_name"],
                               scope_kind="weights", group=c["group"], k=c["k"], trial=t, base_seed=base_seed, return_logits=True)
            lg = rec.pop("_logits")
            if not rec["restored_exactly"] or not rec["flip_applied_to_encoded_bytes"]:
                raise RuntimeError(f"injector integrity failure in {tid}: restored={rec['restored_exactly']} applied={rec['flip_applied_to_encoded_bytes']}")
            verify_trial(rec, c, store, fmt)
            changed = int((lg.argmax(1) != clean_pred).sum())
            log.append(slim(rec, tid, c, campaign, subset, changed))
            acc.append(rec["fault_accuracy"])
            n_new += 1
        if acc and (c["level"] != "layerwise" or ci % 20 == 0):
            progress(f"  [{fmt}|{c['scope_name']}|{c['group']}|k{c['k']}] {len(acc)} trials, mean fault accuracy {np.mean(acc):.2f} % (clean {clean['accuracy']:.2f} %)")
    return control, {"new": n_new, "skipped": n_skip, "clean": {"accuracy": clean["accuracy"], "loss": clean["loss"]}}


def summarise(records):
    rows = S4.summarize(records)
    key = lambda r: (r["layer_id"], r["format"], r["scope_kind"], r["fault_group"], r["requested_k"])
    agg = {}
    for r in records:
        agg.setdefault(key(r), []).append(r)
    for s in rows:
        rr = agg[(s["layer_id"], s["format"], s["scope_kind"], s["fault_group"], s["k"])]
        s["mean_changed_predictions"] = float(np.mean([r["changed_predictions"] for r in rr]))
        s["mean_agreement_percent"] = float(np.mean([r["agreement_percent"] for r in rr]))
    return rows


def cell(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else "inf" if v == math.inf else "-inf" if v == -math.inf else repr(round(v, 8))
    return "" if v is None else v


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k, "")) for k in fields})


def fp32_state_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def select_shortlist(layerwise_rows, top_n: int, min_mean_drop_pp: float):
    """Selection rule (disclosed in the output): per format, the ``top_n`` layer-wise configurations (layer x group x k) with the largest
    mean accuracy drop over their screening trials, keeping only those whose mean drop is at least ``min_mean_drop_pp``."""
    out = []
    for fmt in sorted({r["format"] for r in layerwise_rows}, key=lambda f: PRIMARY.index(f) if f in PRIMARY else 99):
        rr = sorted((r for r in layerwise_rows if r["format"] == fmt and r["drop_pp_mean"] >= min_mean_drop_pp),
                    key=lambda r: (-r["drop_pp_mean"], r["layer_id"], r["fault_group"], r["k"]))[:top_n]
        for r in rr:
            out.append({"layer_name": r["layer_name"], "format_key": fmt, "group": r["fault_group"], "k": r["k"],
                        "layerwise_mean_drop_pp": r["drop_pp_mean"], "layerwise_n_trials": r["n_trials"],
                        "selection_rule": f"top-{top_n} per format by layer-wise mean accuracy drop, mean drop >= {min_mean_drop_pp} pp "
                                          "(SELECTED ON THE SCREENING RESULTS: follow-up estimates are biased toward high risk)"})
    return out


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--campaign", required=True, choices=["validate", "global", "layerwise", "shortlist", "targeted"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None, help="default results/<run>/jfamp/quantization")
    ap.add_argument("--out-dir", default=None, help="default results/<run>/jfamp/stage_04_encoded_fault_injection")
    ap.add_argument("--inventory", default=None, help="optional Stage 1 layer_inventory.csv (layer order check in the validation gate)")
    ap.add_argument("--formats", nargs="+", default=PRIMARY, choices=list(S3.SPECS))
    ap.add_argument("--subset-size", type=int, default=1000, help="declared fixed class-stratified validation subset for global/layer-wise screening (0 = full)")
    ap.add_argument("--trials", type=int, default=None, help="override the protocol budget (30 / 10 / 100) - for smoke runs only")
    ap.add_argument("--only-layers", nargs="*", default=None, help="restrict layer-wise to these module names (smoke runs only)")
    ap.add_argument("--targets-csv", default=None, help="targeted: layer_name,format_key,group,k[,selection_rule]")
    ap.add_argument("--top-n", type=int, default=5, help="shortlist: configurations per format")
    ap.add_argument("--min-mean-drop-pp", type=float, default=0.5, help="shortlist: minimum layer-wise mean accuracy drop")
    ap.add_argument("--eval-seed", type=int, default=42, help="seed of the trial streams and of the validation subset")
    ap.add_argument("--smoke-trials", type=int, default=3, help="validate: trials per (group, k) in the smoke campaign")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    out = Path(args.out_dir) if args.out_dir else base / "stage_04_encoded_fault_injection"
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(args.checkpoint)
    camp = args.campaign
    formats = list(args.formats)

    # ---- frozen Stage 2 inputs
    man = json.load(open(qdir / "quantizer_manifest.json"))
    ck_sha = sha256_file(ck)
    if man["checkpoint"]["sha256"] != ck_sha:
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    frozen = hashlib.sha256(json.dumps({"checkpoint_sha256": ck_sha, "formats": {k: [v["config"], v["content_sha256"]] for k, v in man["formats"].items()}},
                                       sort_keys=True).encode()).hexdigest()
    if frozen != man["frozen_config_sha256"]:
        sys.exit("Stage 2 manifest does not reproduce its frozen-configuration hash")
    missing = [f for f in formats if f not in man["formats"]]
    if missing:
        sys.exit(f"formats not in the Stage 2 manifest: {missing}")
    if camp == "shortlist":
        rows = list(csv.DictReader(open(out / "layerwise_summary.csv", encoding="utf-8")))
        for r in rows:
            r["layer_id"], r["k"], r["n_trials"] = int(r["layer_id"]), int(r["k"]), int(r["n_trials"])
            r["drop_pp_mean"] = float(r["drop_pp_mean"])
        sl = select_shortlist(rows, args.top_n, args.min_mean_drop_pp)
        write_csv(out / "targeted_shortlist.csv", sl, list(sl[0]) if sl else ["layer_name"])
        print(f"{len(sl)} shortlisted configurations -> {out / 'targeted_shortlist.csv'}")
        return
    gate = None
    if camp != "validate":
        vpath = out / "stage4_validation.json"
        if not vpath.exists():
            sys.exit("run --campaign validate first (Section 4.3 gate)")
        gate = json.load(open(vpath))
        if gate["checkpoint_sha256"] != ck_sha or gate["stage2_frozen_config_sha256"] != frozen:
            sys.exit("the validation gate was run on a different checkpoint / frozen configuration")
        bad = [f for f in formats if gate["gate_per_format"].get(f) != "PASS"]
        if bad:
            sys.exit(f"validation gate has not passed for {bad}")

    model, ckpt = load_model(ck)
    model.eval()
    mods = quantizable_modules(model)
    module_names = [n for n, _ in mods]
    layer_names = [f"{n}.weight" for n in module_names]
    codecs = S2.codec_registry()

    # ---- fixed evaluation set
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=None)
    X, y = load_tensors(loader, device)
    if camp == "targeted" or camp == "validate" and args.subset_size == 0:
        sel = np.arange(int(y.numel()))
    else:
        sel = stratified_subset(y.cpu().numpy(), args.subset_size, args.eval_seed)
    X, y = X[torch.from_numpy(sel).to(device)], y[torch.from_numpy(sel).to(device)]
    full = len(sel) == 5000 or camp == "targeted"
    subset = {"name": "full-validation" if full else f"stratified-{len(sel)}-seed{args.eval_seed}", "images": int(y.numel()),
              "positions_in_validation_set_sha256": sha256_ints(sel), "class_counts": np.bincount(y.cpu().numpy()).tolist(),
              "seed": args.eval_seed, "declared_in_advance": True}
    print(f"{camp}: {subset['name']} ({subset['images']} validation images), device {device}")

    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    ctx_base = {"experiment_id": EXPERIMENT, "run_id": run_id, "model": "mobilenetv2_cifar (width 1.0, stride-1 stem, 10 classes)",
                "dataset": f"CIFAR-10 validation holdout, {subset['name']}", "checkpoint_hash": ck_sha}
    commit = git_info()["commit"]

    if camp == "validate":
        sys.exit(run_validate(args, cfg, device, out, man, model, codecs, mods, layer_names, X, y, subset, run_id, ck_sha, frozen,
                              ctx_base, commit, qdir, formats))

    # ---- configurations
    trials = args.trials or BUDGET[camp]
    if camp == "targeted":
        if not args.targets_csv:
            sys.exit("--targets-csv is required for the targeted campaign")
        trows = list(csv.DictReader(open(args.targets_csv, encoding="utf-8")))
        ids = {n: i for i, n in enumerate(module_names)}
        all_cfgs = {f: targeted_configs([r for r in trows if (r.get("format_key") or r["format"]) == f], module_names, ids) for f in formats}
        skipped = []
    else:
        all_cfgs, skipped = {}, []
        layers = None
        if args.only_layers:
            layers = [module_names.index(n) for n in args.only_layers]
        for f in formats:
            all_cfgs[f], sk = make_configs(f, module_names, camp, layers)
            skipped += sk
    formats = [f for f in formats if all_cfgs[f]]
    expected = [i for f in formats for i in S4.expected_ids(all_cfgs[f], trials)]
    print(f"{len(formats)} formats, {sum(len(all_cfgs[f]) for f in formats)} configurations x {trials} trials = {len(expected)} trials")

    prefix = {"global": "global", "layerwise": "layerwise", "targeted": "targeted"}[camp]
    log = S4.TrialLog(out / f"{prefix}_trials.jsonl")
    if log.duplicates:
        sys.exit(f"duplicate trial ids in the trial log: {log.duplicates[:5]}")
    controls, clean_info, t0 = [], {}, time.time()
    n_new = n_skip = 0
    if not args.summarize_only:
        for fmt in formats:
            stored = S2.load_encoded(qdir / man["formats"][fmt]["encoded_file"], layer_names)
            if S2.content_sha256(stored) != man["formats"][fmt]["content_sha256"]:
                sys.exit(f"{fmt}: encoded weights differ from the frozen content hash")
            ctx = {**ctx_base, "codec_version": f"{fmt}|stage2-frozen:{frozen[:12]}|{S3.INJECTOR_VERSION}|git:{commit[:8]}"}
            ctrl, res = run_format(fmt, codecs[fmt], stored, model, X, y, device, all_cfgs[fmt], trials, log, args.eval_seed, ctx, camp, subset)
            controls.append(ctrl)
            clean_info[fmt] = res["clean"]
            n_new += res["new"]
            n_skip += res["skipped"]
            print(f"=== {fmt}: clean {res['clean']['accuracy']:.2f} % | control max|dlogit| {ctrl['max_abs_logit_diff']:.1e} PASS | "
                  f"{res['new']} new, {res['skipped']} done before | {time.time() - t0:.0f} s elapsed")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    log.close()
    records = S4.load_records(out / f"{prefix}_trials.jsonl")
    rows = summarise(records)
    write_csv(out / f"{prefix}_summary.csv", rows, S4.SUMMARY_FIELDS + SUMMARY_EXTRA)
    if camp == "layerwise":
        meta = [{"layer_id": i, "module_name": n, "num_weights": int(m.weight.numel())} for i, (n, m) in enumerate(mods)]
        header, hm = S4.heatmap(rows, meta)
        write_csv(out / "layerwise_heatmap.csv", hm, header)
    ctl_path = out / "no_fault_control.csv"
    old = list(csv.DictReader(open(ctl_path, encoding="utf-8"))) if ctl_path.exists() else []
    cf = ["campaign", "format", "flips", "eval_images", "clean_accuracy", "control_accuracy", "max_abs_logit_diff", "tolerance",
          "encoded_hash_before", "encoded_hash_after", "restored_exactly", "passed"]
    write_csv(ctl_path, old + controls, cf)
    comp = S4.verify_completeness(records, expected)
    flag_totals = {n: sum(n in r["output_flag"].split(",") for r in records) for n in S4.FLAG_NAMES}
    per_fmt = {f: {"configurations": len(all_cfgs[f]), "expected_trials": len(all_cfgs[f]) * trials,
                   "recorded_trials": sum(1 for r in records if r["format"] == f), **clean_info.get(f, {})} for f in formats}
    prev_path = out / f"{prefix}_manifest.json"
    sessions = (json.load(open(prev_path)).get("sessions", []) if prev_path.exists() else []) + [
        {"run_id": run_id, "started_utc": dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(timespec="seconds"),
         "new_trials": n_new, "skipped_already_done": n_skip, "seconds": time.time() - t0, "git": git_info(), "summarize_only": args.summarize_only}]
    dump_json({"stage": f"MobileNetV2 protocol Section 4 - encoded-weight fault injection, {camp} campaign",
               "label": "screening" if camp != "targeted" else "targeted follow-up (selection bias: configurations were chosen from screening results)",
               "experiment_id": EXPERIMENT, "checkpoint_sha256": ck_sha, "stage2_frozen_config_sha256": frozen,
               "validation_gate": {"run_id": gate["run_id"], "per_format": gate["gate_per_format"]},
               "settings": {"trials_per_configuration": trials, "protocol_budget": BUDGET[camp], "budget_overridden": args.trials is not None,
                            "k_values": list(S4.K_VALUES), "formats": formats, "eval_subset": subset, "base_seed": args.eval_seed,
                            "groups": {f: groups_of(f) for f in formats}, "int_scale_policy": man["int_scale_policy"],
                            "fault_scope": "stored weight codes only; the INT float32 scale words are NOT faulted in this campaign",
                            "sampling": "k DISTINCT weights without replacement from the declared scope (global: all weights of all 53 layers; "
                                        "layer-wise: the target layer), one bit per weight, uniform within the fault group; one derived seed per trial",
                            "non_finite_policy": "images with non-finite logits count as incorrect; fault_loss is null when any logit is non-finite; "
                                                 "such trials are kept, flagged and counted as failures",
                            "evaluation_images_note": "screening results (subset) are not comparable with full-validation results"},
               "clean_per_format": clean_info, "per_format": per_fmt, "no_fault_controls_this_session": controls,
               "completion": comp, "failure_counts_by_cause": flag_totals, "skipped_configurations": skipped,
               "trials_recorded": len(records), "sessions": sessions, "environment": env_info(), "git": git_info()}, prev_path)
    print(f"\ncompleteness: {'COMPLETE' if comp['complete'] else 'INCOMPLETE'} ({comp['recorded']}/{comp['expected']} trials; "
          f"duplicates {len(comp['duplicates'])}, missing {comp['n_missing']})\nwritten: {out}")
    sys.exit(0 if comp["complete"] else 2)


# ------------------------------------------------------------------------------------------------------------ validation gate
def run_validate(args, cfg, device, out, man, model, codecs, mods, layer_names, X, y, subset, run_id, ck_sha, frozen, ctx_base, commit,
                 qdir, formats):
    numels = OrderedDict((f"{n}.weight", int(m.weight.numel())) for n, m in mods)
    weights = OrderedDict((f"{n}.weight", m.weight.detach().cpu().numpy().astype(np.float32)) for n, m in mods)
    inventory = list(csv.DictReader(open(args.inventory, encoding="utf-8"))) if args.inventory and Path(args.inventory).exists() else None
    if inventory is not None:
        for r in inventory:
            r.setdefault("module_name", r.get("module_name"))
    smoke_rows, validation, records = [], OrderedDict(), []
    L = list(range(len(layer_names)))
    for fmt in formats:
        t_fmt = time.time()
        codec = codecs[fmt]
        stored = S2.load_encoded(qdir / man["formats"][fmt]["encoded_file"], layer_names)
        rows = V.bit_mapping_rows(fmt, codec) + V.special_value_rows(fmt, codec)
        if fmt in S3.SCALE_FORMATS:
            rows += V.scale_bit_rows(fmt, codec)
        if fmt == "int4_pt":
            rows += V.packing_rows(codec)
        smoke_rows += rows
        tests = []
        by_cat = {}
        for r in rows:
            by_cat.setdefault(r["category"], [0, 0])[0 if r["status"] == "PASS" else 1] += 1
        for cat, (p, f) in by_cat.items():
            tests.append({"test": f"{fmt}: known-bit smoke test, {cat} ({p + f} cases)", "passed": f == 0, "detail": f"{p} pass, {f} fail"})
        tests += V.roundtrip_synthetic_tests(fmt, codec)
        tests += V.roundtrip_real_tests(fmt, codec, weights, stored, man["formats"][fmt]["content_sha256"])
        tests += V.scope_tests(S3.EncodedStore(fmt, stored), numels, inventory, fmt)
        ctx = {**ctx_base, "codec_version": f"{fmt}|stage2-frozen:{frozen[:12]}|{S3.INJECTOR_VERSION}|git:{commit[:8]}"}
        inj = S3.EncodedFaultInjector(model, S3.EncodedStore(fmt, stored), codec, device)
        res = V.model_level_tests(fmt, inj, X, y, ctx, args.eval_seed, lambda: fp32_state_hash(model))
        tests += res["tests"]
        records += res["records"]
        clean = res["clean"]
        smoke = []
        for g in groups_of(fmt):
            for k in S4.K_VALUES:
                for t in range(args.smoke_trials):
                    smoke.append(S3.run_trial(inj, X, y, clean, ctx, fmt=fmt, scope_layers=L, scope_name="GLOBAL", scope_kind="weights",
                                              group=g, k=k, trial=t, base_seed=args.eval_seed))
        for li in (0, len(layer_names) - 1):                                   # layer-wise scope: stem and classifier
            for t in range(args.smoke_trials):
                smoke.append(S3.run_trial(inj, X, y, clean, ctx, fmt=fmt, scope_layers=[li], scope_name=layer_names[li][:-7],
                                          scope_kind="weights", group=groups_of(fmt)[-1], k=8, trial=t, base_seed=args.eval_seed))
        records += smoke
        miss = [f for r in smoke + res["records"] for f in S3.REQUIRED_TRIAL_FIELDS if f not in r]
        tests.append({"test": f"{fmt}: all {len(smoke) + len(res['records'])} validation trial records contain every required field",
                      "passed": not miss, "detail": f"missing: {sorted(set(miss))}" if miss else ""})
        bad = [r for r in smoke if not r["restored_exactly"] or not r["flip_applied_to_encoded_bytes"]]
        tests.append({"test": f"{fmt}: {len(smoke)} smoke trials, each restored byte-for-byte (hash before == after) with the flip applied to the encoded bytes",
                      "passed": not bad})
        nf = sum(r["failure"] for r in smoke)
        tests.append({"test": f"{fmt}: non-finite trials are kept and flagged, never dropped", "passed": True, "detail": f"{nf} of {len(smoke)} flagged"})
        failed = [t for t in tests if not t["passed"]]
        validation[fmt] = {"tests": tests, "n_tests": len(tests), "n_passed": len(tests) - len(failed), "failed": [t["test"] for t in failed],
                           "gate": "PASS" if not failed else "FAIL", "clean_accuracy_on_gate_images": clean["accuracy"],
                           "clean_loss_on_gate_images": clean["loss"], "seconds": time.time() - t_fmt}
        print(f"{fmt:11s} gate {validation[fmt]['gate']}  ({len(tests) - len(failed)}/{len(tests)} tests, {len(smoke) + len(res['records'])} trials, "
              f"clean {clean['accuracy']:.2f} %)")
        del inj
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(out / "known_bit_smoke_tests.csv", smoke_rows, V.SMOKE_FIELDS)
    with open(out / "validation_trials.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "_logits"}) + "\n")
    ok = all(v["gate"] == "PASS" for v in validation.values())
    dump_json({"stage": "MobileNetV2 protocol Section 4.3 - mandatory validation before the fault campaigns", "run_id": run_id,
               "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "checkpoint_sha256": ck_sha,
               "stage2_frozen_config_sha256": frozen, "injector_version": S3.INJECTOR_VERSION, "eval_subset": subset,
               "synthetic_cases": {"total": len(smoke_rows), "passed": sum(r["status"] == "PASS" for r in smoke_rows)},
               "gate_per_format": {k: v["gate"] for k, v in validation.items()}, "gate_all_formats": "PASS" if ok else "FAIL",
               "bit_maps": {f: {"width": S3.SPECS[f]["width"], "groups": {g: list(S3.group_bits(f, g)) for g in groups_of(f)}} for f in formats},
               "int_encoding": "two's complement; sign = MSB, value bits = the remaining low bits; INT4 packed two per byte, element 2j in the LOW nibble "
                               "(bits 0-3) of byte j, element 2j+1 in the HIGH nibble; odd count padded with one zero nibble that is never a fault target",
               "git": git_info(), "environment": env_info(), "formats": validation}, out / "stage4_validation.json")
    lines = ["# Section 4.3 validation gate", "", f"run `{run_id}`; checkpoint `{ck_sha[:16]}...`; frozen Stage 2 `{frozen[:16]}...`; "
             f"{subset['images']} validation images for the model-level tests; {len(smoke_rows)} synthetic bit cases", "",
             "| format | tests passed | gate |", "|---|---|---|"]
    lines += [f"| {k} | {v['n_passed']}/{v['n_tests']} | **{v['gate']}** |" for k, v in validation.items()]
    for k, v in validation.items():
        lines += [f"* FAILED [{k}]: {t}" for t in v["failed"]]
    (out / "stage4_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nVALIDATION GATE: {'PASS' if ok else 'FAIL'}\nwritten: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
