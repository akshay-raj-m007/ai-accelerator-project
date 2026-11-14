#!/usr/bin/env python
"""MobileNetV2 protocol Section 6, part 2 (GPU machine): assemble the assignments, MEASURE them end to end, freeze one, evaluate the test set once.

For every distinct optimal assignment of stage6_optimize.py (plus uniform single-datatype references and uniform protected references):
  A. the assembled model (each layer decoded from its own frozen encoding) is evaluated on all 5,000 validation images -> A_clean(x), loss, actual stored bytes
  B. a shortlist (non-dominated fronts of the datatype-only and of the protected assignments, plus the references) gets end-to-end fault trials on the STORED bytes of
     the assembled model, 1,000-image class-stratified validation subset (the Section 4 subset): faults are drawn on the stored bits, passed through every layer's real
     protection decoder (parity / SECDED / DMR / scale TMR), and the RESIDUAL flips are applied to the stored data bytes -> measured R_fault (weights_k8) and the
     scale-word metric (scales_k1)
  C. the constraint grid (delta_clean, tau_fault, B, tau_scale): the best MEASURED-feasible assignment per setting (objective = lambda-weighted Qn + Rn of the table)
  D. with --finalize: the declared operating point (defaults = declared_constants.json) selects ONE assignment; assignment.json is written and frozen from validation
     data; more validation fault trials (weights_k8, stored_k8 incl. redundancy bits, scales_k1) are run; only then the official test set is evaluated (once).

Declared fault distributions (weight-only quantization; biases / BatchNorm stay FP32 and are never faulted):
  weights_k8  PRIMARY. 8 distinct weights over all quantizable weights, one bit uniform over all bits of the weight's own encoding (Stage 4 'any bit', k = 8).
  stored_k8   SUPPLEMENTARY (final assignment). 8 distinct stored bits uniform over all stored payload bits INCLUDING the parity / SECDED check bits and the DMR copy.
  scales_k1   SECOND metric, never merged. 1 flip uniform over all stored scale bits (32 per unprotected INT layer, 3 x 32 scale TMR, 32 + redundancy word-protected).

  python -u scripts/stage6_end_to_end.py --checkpoint $CK --run-name mobilenetv2_cifar10                       # A-C
  python -u scripts/stage6_end_to_end.py --checkpoint $CK --run-name mobilenetv2_cifar10 --finalize            # D (after A-C finished)
"""
import argparse
import csv
import datetime as dt
import json
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from fr import data as fdata
from fr import ecc
from fr import stage2 as S2
from fr import stage3 as S3
from fr import stage4 as S4
from fr import stage5_e2e as E2
from fr import stage6 as S6
from fr.baseline_profile import sha256_file, sha256_ints
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.layer_scores import spearman
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, env_info, git_info, resolve_device
from stage4_encoded_fault_injection import stratified_subset

EXPERIMENT_ID = "jfamp-mobilenetv2-cifar10-stage6-end-to-end"
MODEL = "mobilenetv2_cifar (width 1.0, stride-1 stem, 10 classes)"


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def has_scale_words(tokens) -> bool:
    return any(S3.SPECS[S6.parse_option(t)[0]]["kind"] == "int" for t in tokens)


def predicted_scale_drop(p, choice) -> float:
    """Expected drop (pp) of scales_k1 from the layer table: stored-scale-bit-weighted mean of the layer scale-word risks."""
    bits = []
    for c in choice:
        f, pr = S6.parse_option(p.formats[c])
        if S3.SPECS[f]["kind"] != "int":
            bits.append(0)
        else:
            bits.append(96 if pr == S6.TMR else ecc.STORED_BITS[pr])
    bits = np.array(bits, dtype=float)
    return 0.0 if bits.sum() == 0 else float((bits / bits.sum() * p.Rscale[np.arange(len(choice)), list(choice)]).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--opt-dir", default=None, help="folder of stage6_optimize.py outputs (default results/<run>/jfamp/stage_06_optimization)")
    ap.add_argument("--n-fault-images", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=100, help="independent fault trials per assignment and distribution (shortlist)")
    ap.add_argument("--max-fault-assignments", type=int, default=24)
    ap.add_argument("--include", nargs="*", default=[], help="extra assignment ids that must get fault trials")
    ap.add_argument("--lambda-", dest="lam", type=float, default=1.0)
    ap.add_argument("--norm", default="minmax")
    ap.add_argument("--delta-clean", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--tau-fault", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--tau-scale", type=float, nargs="+", default=[1.0])
    ap.add_argument("--avg-bits", type=float, nargs="+", default=[4.8, 5, 6, 8, 10, 12, 16])
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--final-delta-clean", type=float, default=None, help="default: declared_constants.json")
    ap.add_argument("--final-tau-fault", type=float, default=None)
    ap.add_argument("--final-avg-bits", type=float, default=None)
    ap.add_argument("--final-tau-scale", type=float, default=None)
    ap.add_argument("--final-trials", type=int, default=300)
    ap.add_argument("--test-fault-trials", type=int, default=100)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--test-images", type=int, default=None, help="dry runs only")
    ap.add_argument("--val-images", type=int, default=None, help="dry runs only: limit the clean validation set")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    odir = Path(args.opt_dir) if args.opt_dir else base / "stage_06_optimization"
    declared = json.load(open(odir / "declared_constants.json"))
    fin = {"delta_clean": args.final_delta_clean if args.final_delta_clean is not None else declared["delta_clean_pp"],
           "tau_fault": args.final_tau_fault if args.final_tau_fault is not None else declared["tau_fault_pp"],
           "avg_bits": args.final_avg_bits if args.final_avg_bits is not None else declared["B_avg_bits_per_weight"],
           "tau_scale": args.final_tau_scale if args.final_tau_scale is not None else declared["tau_scale_pp"]}
    table_rows = list(csv.DictReader(open(odir / "jfamp_candidate_table.csv", encoding="utf-8")))
    p, nw = S6.rows_to_problem(table_rows, "joint")
    L, N = len(p.layers), float(nw.sum())
    asg = OrderedDict()
    for r in csv.DictReader(open(odir / "distinct_assignments.csv", encoding="utf-8")):
        asg[r["assignment_id"]] = {"options": [S6.long_token(s) for s in r["assignment"].split(";")], "n_settings": int(r["n_settings"]), "modes": r["found_in_modes"]}
    for aid, opts in S6.uniform_assignments(L).items():
        asg[aid] = {"options": opts, "n_settings": 0, "modes": "reference"}
    for a in asg.values():
        assert len(a["options"]) == L and all(o in p.formats for o in a["options"])
    choice_of = {k: [p.formats.index(o) for o in v["options"]] for k, v in asg.items()}
    pred = {k: p.totals(c) for k, c in choice_of.items()}
    pred_scale = {k: predicted_scale_drop(p, c) for k, c in choice_of.items()}
    man2 = json.load(open(qdir / "quantizer_manifest.json"))
    ck = Path(args.checkpoint)
    ck_sha = sha256_file(ck)
    if man2["checkpoint"]["sha256"] != ck_sha:
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    frozen = man2["frozen_config_sha256"]
    meta = json.load(open(odir / "candidate_table_meta.json"))
    if meta["checkpoint_sha256"] != ck_sha or meta["stage2_frozen_config_sha256"] != frozen:
        sys.exit("the candidate table was made for another checkpoint / frozen configuration")
    clean_path, trials_path, manifest_path = odir / "end_to_end_clean.csv", odir / "end_to_end_fault_trials.jsonl", odir / "end_to_end_manifest.json"
    prev = json.load(open(manifest_path)) if manifest_path.exists() else {}
    run_id = prev.get("run_id") or (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    clean_rows = {r["assignment_id"]: r for r in csv.DictReader(open(clean_path))} if clean_path.exists() else {}
    fp32, shortlist, t0 = prev.get("fp32_reference"), prev.get("shortlist", []), time.time()

    def load_stores(lnames):
        stores = {}
        for f in S6.FORMATS:
            enc = S2.load_encoded(qdir / man2["formats"][f]["encoded_file"], lnames)
            if S2.content_sha256(enc) != man2["formats"][f]["content_sha256"]:
                sys.exit(f"{f}: encoded weights differ from the frozen content hash")
            stores[f] = S3.EncodedStore(f, enc)
        return stores

    def load_data(model, ckpt):
        norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
        val_size, split_seed = split_params(ckpt, cfg)
        Xv, yv = load_tensors(fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=args.val_images), device)
        sel = stratified_subset(yv.cpu().numpy(), args.n_fault_images, args.eval_seed)
        idx = torch.from_numpy(sel).to(device)
        return norm, Xv, yv, Xv[idx], yv[idx], {"name": f"stratified-{len(sel)}-seed{args.eval_seed}" if len(sel) < int(yv.numel()) else "full-validation", "images": int(len(sel)),
                                                  "positions_in_validation_set_sha256": sha256_ints(sel), "seed": args.eval_seed}

    codecs = S2.codec_registry()

    def assemble(model, stores, aid, opts):
        ms = E2.MixedStore([S6.base_token(t) for t in opts], stores)
        prots = [S6.parse_option(t)[1] for t in opts]
        return ms, E2.MixedFaultInjector(model, ms, codecs, device), S6.ProtectedFaults(ms, prots), prots

    if not args.summarize_only:
        model, ckpt = load_model(ck)
        model.eval()
        mods = quantizable_modules(model)
        lnames = [f"{n}.weight" for n, _ in mods]
        if [n for n, _ in mods] != p.layers:
            sys.exit("layer order differs from the candidate table")
        norm, Xv, yv, Xf, yf, subset = load_data(model, ckpt)
        stores = load_stores(lnames)
        ctx = {"experiment_id": EXPERIMENT_ID, "run_id": run_id, "model": MODEL, "dataset": f"CIFAR-10 validation holdout, {subset['name']}", "checkpoint_hash": ck_sha,
               "codec_version": f"mixed|stage2-frozen:{frozen[:12]}|{S3.INJECTOR_VERSION}|{S6.SCHEMA}"}
        print(f"run {run_id}: {int(yv.numel())} validation images (clean), {subset['images']} for fault trials ({subset['name']}), device {device}")
        if fp32 is None:
            ev = S3.evaluate(model.to(device), Xv, yv)
            fp32 = {"val_accuracy": ev["accuracy"], "val_loss": ev["loss"], "val_samples": int(yv.numel())}
        print(f"FP32 reference: {fp32['val_accuracy']:.2f} % / loss {fp32['val_loss']:.4f} on {fp32['val_samples']} validation images")

        # ---- A. clean measurements
        n_new = 0
        for aid in asg:
            if aid in clean_rows:
                continue
            ms, inj, pf, prots = assemble(model, stores, aid, asg[aid]["options"])
            ev = S3.evaluate(inj.model, Xv, yv)
            sb = S6.stored_bytes(ms, prots)
            assert abs(sb["total_bytes"] - pred[aid]["storage_bytes"]) < 1e-6, f"{aid}: stored bytes {sb['total_bytes']} != table {pred[aid]['storage_bytes']}"
            clean_rows[aid] = {"assignment_id": aid, "modes": asg[aid]["modes"], "n_settings": asg[aid]["n_settings"], "val_samples": ev["total"], "clean_val_accuracy": ev["accuracy"],
                               "clean_val_loss": ev["loss"], "clean_drop_pp_vs_fp32": fp32["val_accuracy"] - ev["accuracy"], "delta_ce_vs_fp32": ev["loss"] - fp32["val_loss"],
                               "payload_bytes": sb["payload_bytes"], "scale_bytes": sb["scale_bytes"], "alignment_padding_bytes": sb["alignment_padding_bytes"],
                               "redundancy_bytes": sb["redundancy_bytes"], "storage_bytes": sb["total_bytes"], "avg_bits_per_weight": 8 * sb["total_bytes"] / N,
                               "pred_sum_Q": pred[aid]["sum_Q"], "pred_weight_fault_drop_pp": pred[aid]["sum_R_pp"], "pred_scale_word_drop_pp": pred_scale[aid],
                               "pred_max_layer_scale_risk_pp": pred[aid]["max_layer_scale_risk_pp"], "n_protected_layers": sum(1 for o in asg[aid]["options"] if "+" in o),
                               "encoded_hash": ms.hash(), "assignment": ";".join(S6.short_token(o) for o in asg[aid]["options"])}
            n_new += 1
            del inj
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if n_new % 10 == 0:
                write_csv(clean_path, list(clean_rows.values()))
                print(f"  clean: {len(clean_rows)}/{len(asg)} assignments", flush=True)
        write_csv(clean_path, [clean_rows[a] for a in asg])
        print(f"A. clean measurements: {len(asg)} assignments ({n_new} new)")

        # ---- B/C. shortlist and end-to-end fault trials
        if not shortlist:
            def front_for(ids):
                pts = [[float(clean_rows[a]["storage_bytes"]), float(clean_rows[a]["pred_weight_fault_drop_pp"]), float(clean_rows[a]["pred_max_layer_scale_risk_pp"]),
                        float(clean_rows[a]["clean_drop_pp_vs_fp32"])] for a in ids]
                fr = [ids[i] for i in E2.pareto_front(pts)]
                fr.sort(key=lambda a: float(clean_rows[a]["storage_bytes"]))
                half = max(1, args.max_fault_assignments // 2)
                if len(fr) > half:
                    sel = np.unique(np.round(np.linspace(0, len(fr) - 1, half)).astype(int))
                    fr = [fr[i] for i in sel]
                return fr
            ids = [a for a in asg if asg[a]["modes"] != "reference"]
            prot_ids = [a for a in ids if any("+" in o for o in asg[a]["options"])]
            plain_ids = [a for a in ids if a not in prot_ids]
            front = (front_for(plain_ids) if plain_ids else []) + (front_for(prot_ids) if prot_ids else [])
            shortlist = front + [a for a in args.include if a in asg and a not in front] + [a for a in asg if asg[a]["modes"] == "reference"]
        print(f"B. fault shortlist ({len(shortlist)}): {shortlist}")
        log = S4.TrialLog(trials_path)
        if log.duplicates:
            sys.exit(f"duplicate trial ids in {trials_path}")
        n_trials_new = 0
        for aid in shortlist:
            dists = [d for d in ("weights_k8", "scales_k1") if d != "scales_k1" or has_scale_words(asg[aid]["options"])]
            need = [(d, t) for d in dists for t in range(args.trials) if not log.has(f"{aid}|{d}|k{S6.DIST_K[d]}|t{t}")]
            if not need:
                continue
            ms, inj, pf, prots = assemble(model, stores, aid, asg[aid]["options"])
            clean = S3.evaluate(inj.model, Xf, yf)
            t1 = time.time()
            for d, t in need:
                rec = S6.run_protected_trial(inj, pf, Xf, yf, clean, ctx, assignment_id=aid, dist=d, trial=t, base_seed=args.eval_seed)
                if not rec["restored_exactly"] or not rec["flip_applied_to_encoded_bytes"]:
                    sys.exit(f"{aid}: restoration / flip check failed")
                log.append(rec)
                n_trials_new += 1
            print(f"  {aid}: {len(need)} trials in {time.time() - t1:.0f} s (clean {clean['accuracy']:.2f} % on {int(yf.numel())} images)", flush=True)
            del inj
            if device.type == "cuda":
                torch.cuda.empty_cache()
        log.close()
        print(f"C. fault trials: {n_trials_new} new")
        dump_json({"run_id": run_id, "fp32_reference": fp32, "shortlist": shortlist, "checkpoint_sha256": ck_sha, "stage2_frozen_config_sha256": frozen, "eval_subset": subset,
                   "settings": {"trials": args.trials, "n_fault_images": args.n_fault_images, "base_seed": args.eval_seed, "distributions": list(S6.DISTRIBUTIONS),
                                "lambda": args.lam, "norm": args.norm}, "git": git_info(), "environment": env_info(), "seconds_last_session": time.time() - t0}, manifest_path)
        prev = json.load(open(manifest_path))

    # ======================================================================== summaries (saved CSV / JSONL only)
    clean_rows = {r["assignment_id"]: r for r in csv.DictReader(open(clean_path))}
    fp32, shortlist = prev["fp32_reference"], prev["shortlist"]
    recs = S4.load_records(trials_path) if trials_path.exists() else []
    by = {}
    for r in recs:
        by.setdefault((r["assignment_id"], r["distribution"]), []).append(r)
    trials_expected = prev["settings"]["trials"]
    problems = [f"{a}/{d}: {len(rs)} trials" for (a, d), rs in by.items() if len(rs) != trials_expected or len({r['trial_id'] for r in rs}) != len(rs)]
    items = []
    for aid in asg:
        cr, opts = clean_rows[aid], asg[aid]["options"]
        items.append({"assignment_id": aid, "modes": asg[aid]["modes"], "options": opts, "n_protected_layers": int(cr["n_protected_layers"]),
                      "storage": {"actual_stored_bytes": int(cr["storage_bytes"]), "predicted_from_table_bytes": pred[aid]["storage_bytes"], "avg_bits_per_weight": float(cr["avg_bits_per_weight"]),
                                  "payload_bytes": int(cr["payload_bytes"]), "scale_bytes": int(cr["scale_bytes"]), "alignment_padding_bytes": int(cr["alignment_padding_bytes"]),
                                  "redundancy_bytes": int(cr["redundancy_bytes"])},
                      "clean_validation": {"samples": int(cr["val_samples"]), "accuracy": float(cr["clean_val_accuracy"]), "drop_pp_vs_fp32": float(cr["clean_drop_pp_vs_fp32"]),
                                           "loss": float(cr["clean_val_loss"]), "delta_ce_vs_fp32": float(cr["delta_ce_vs_fp32"])},
                      "predicted_layer_additive": {"sum_Q": pred[aid]["sum_Q"], "weight_fault_drop_pp": pred[aid]["sum_R_pp"], "scale_word_drop_pp": pred_scale[aid],
                                                   "max_layer_scale_risk_pp": pred[aid]["max_layer_scale_risk_pp"]},
                      "objective_value": S6.objective_of(p, choice_of[aid], args.lam, args.norm),
                      "fault_end_to_end": {d: E2.summarize_trials(by[(aid, d)]) for d in S6.DISTRIBUTIONS if (aid, d) in by}})
    flat = []
    for it in items:
        row = {"assignment_id": it["assignment_id"], "modes": it["modes"], "n_protected_layers": it["n_protected_layers"], "avg_bits": it["storage"]["avg_bits_per_weight"],
              "storage_bytes": it["storage"]["actual_stored_bytes"], "clean_val_accuracy": it["clean_validation"]["accuracy"], "clean_drop_pp": it["clean_validation"]["drop_pp_vs_fp32"],
              "delta_ce": it["clean_validation"]["delta_ce_vs_fp32"], "pred_sum_Q": it["predicted_layer_additive"]["sum_Q"],
              "pred_weight_fault_drop_pp": it["predicted_layer_additive"]["weight_fault_drop_pp"], "pred_scale_word_drop_pp": it["predicted_layer_additive"]["scale_word_drop_pp"],
              "objective": it["objective_value"]}
        for d in ("weights_k8", "scales_k1"):
            s = it["fault_end_to_end"].get(d)
            for k_ in ("n_trials", "sample_count", "drop_pp_mean", "drop_pp_median", "drop_pp_p90", "drop_pp_worst", "failure_rate"):
                row[f"{d}_{k_}"] = "" if not s else s[k_]
        row["assignment"] = ";".join(S6.short_token(o) for o in it["options"])
        flat.append(row)
    write_csv(odir / "end_to_end_assignments.csv", flat)
    meas = [it for it in items if "weights_k8" in it["fault_end_to_end"]]
    comp = {"note": "layer-additive estimates vs end-to-end measurements of the assembled models", "n_assignments_clean": len(items), "n_assignments_fault": len(meas)}
    if len(items) > 2:
        comp["spearman_predicted_Q_vs_measured_delta_ce"] = spearman([i["predicted_layer_additive"]["sum_Q"] for i in items], [i["clean_validation"]["delta_ce_vs_fp32"] for i in items])
        comp["spearman_predicted_Q_vs_measured_clean_drop_pp"] = spearman([i["predicted_layer_additive"]["sum_Q"] for i in items], [i["clean_validation"]["drop_pp_vs_fp32"] for i in items])
        comp["sum_Q_total_vs_measured_delta_ce"] = [{"assignment_id": i["assignment_id"], "predicted_sum_Q": i["predicted_layer_additive"]["sum_Q"],
                                                     "measured_delta_ce": i["clean_validation"]["delta_ce_vs_fp32"], "measured_clean_drop_pp": i["clean_validation"]["drop_pp_vs_fp32"]} for i in items]
    if len(meas) > 2:
        pw = [i["predicted_layer_additive"]["weight_fault_drop_pp"] for i in meas]
        mw = [i["fault_end_to_end"]["weights_k8"]["drop_pp_mean"] for i in meas]
        comp["spearman_predicted_weight_fault_drop_vs_measured_weights_k8"] = spearman(pw, mw)
        comp["weights_k8_mean_predicted_pp"], comp["weights_k8_mean_measured_pp"] = float(np.mean(pw)), float(np.mean(mw))
        comp["per_assignment"] = [{"assignment_id": i["assignment_id"], "predicted_weight_fault_drop_pp": i["predicted_layer_additive"]["weight_fault_drop_pp"],
                                   "measured_weights_k8_drop_pp_mean": i["fault_end_to_end"]["weights_k8"]["drop_pp_mean"],
                                   "measured_weights_k8_drop_pp_p90": i["fault_end_to_end"]["weights_k8"]["drop_pp_p90"],
                                   "predicted_scale_word_drop_pp": i["predicted_layer_additive"]["scale_word_drop_pp"],
                                   "measured_scales_k1_drop_pp_mean": i["fault_end_to_end"].get("scales_k1", {}).get("drop_pp_mean")} for i in meas]
    cands = [it for it in items if it["modes"] != "reference"]                    # selection is among the optimiser's own assignments; references are comparison only
    grid = []
    for tau_s in [None] + list(args.tau_scale):
        for dlt in args.delta_clean:
            for tau in args.tau_fault:
                for bits in args.avg_bits:
                    f = S6.feasible_items(cands, dlt, tau, bits, tau_s)
                    b = S6.best_item(cands, dlt, tau, bits, tau_s)
                    grid.append({"tau_scale_pp": "" if tau_s is None else tau_s, "delta_clean_pp": dlt, "tau_fault_pp": tau, "B_avg_bits": bits, "n_feasible": len(f),
                                 "selected_assignment": b["assignment_id"] if b else "none feasible", "objective": "" if not b else b["objective_value"],
                                 "avg_bits": "" if not b else b["storage"]["avg_bits_per_weight"], "clean_drop_pp": "" if not b else b["clean_validation"]["drop_pp_vs_fp32"],
                                 "R_fault_weights_k8_pp": "" if not b else b["fault_end_to_end"]["weights_k8"]["drop_pp_mean"],
                                 "scales_k1_pp": "" if not b else (b["fault_end_to_end"].get("scales_k1", {}).get("drop_pp_mean", 0.0)), "n_protected_layers": "" if not b else b["n_protected_layers"]})
    write_csv(odir / "constraint_grid_selection.csv", grid)
    dump_json({"label": "Measured end to end on assembled models (validation data only unless final_test_evaluation.json exists).", "fp32_reference": fp32,
               "protocol": {"clean": "all 5,000 validation images", "fault": f"{prev['eval_subset']['name']} ({prev['eval_subset']['images']} images), {trials_expected} trials per distribution",
                            "distributions": "weights_k8 (PRIMARY), scales_k1 (SECOND metric); stored_k8 (incl. redundancy bits) only for the frozen assignment"},
               "fault_shortlist_rule": "non-dominated front (storage, predicted weight-fault drop, largest scale-word risk, measured clean drop) of the datatype-only assignments and of the "
                                       "protected assignments, each evenly thinned to half the size limit, plus the references", "shortlist": shortlist, "incomplete_cells": problems,
               "predicted_vs_measured": comp, "assignments": items}, odir / "end_to_end_assignment_metrics.json")
    print(f"summaries written to {odir}  (incomplete trial cells: {len(problems)})")

    # ======================================================================== finalize
    if args.finalize:
        if (odir / "final_test_evaluation.json").exists():
            sys.exit("final_test_evaluation.json exists: the official test set is evaluated ONCE; refusing to repeat it")
        b = S6.best_item(cands, fin["delta_clean"], fin["tau_fault"], fin["avg_bits"], fin["tau_scale"])
        if b is None:
            sys.exit("no measured-feasible assignment at the declared operating point; see constraint_grid_selection.csv (nothing was frozen, the test set was not read)")
        aid, opts = b["assignment_id"], b["options"]
        model, ckpt = load_model(ck)
        model.eval()
        mods = quantizable_modules(model)
        lnames = [f"{n}.weight" for n, _ in mods]
        norm, Xv, yv, Xf, yf, subset = load_data(model, ckpt)
        stores = load_stores(lnames)
        ms, inj, pf, prots = assemble(model, stores, aid, opts)
        clean_f = S3.evaluate(inj.model, Xf, yf)
        ctx_v = {"experiment_id": EXPERIMENT_ID + "-final-validation", "run_id": run_id, "model": MODEL, "dataset": f"CIFAR-10 validation holdout, {subset['name']}",
                 "checkpoint_hash": ck_sha, "codec_version": f"mixed|stage2-frozen:{frozen[:12]}|{S6.SCHEMA}"}
        dists = [d for d in S6.DISTRIBUTIONS if d != "scales_k1" or has_scale_words(opts)]
        final_v = {}
        with open(odir / "final_validation_fault_trials.jsonl", "w", encoding="utf-8") as fh:
            for d in dists:
                rs = [S6.run_protected_trial(inj, pf, Xf, yf, clean_f, ctx_v, assignment_id=f"FINAL_{aid}", dist=d, trial=t, base_seed=args.eval_seed) for t in range(args.final_trials)]
                for r in rs:
                    fh.write(json.dumps(r) + "\n")
                final_v[d] = E2.summarize_trials(rs)
                final_v[d]["all_restored_exactly"] = all(r["restored_exactly"] for r in rs)
        sel_json = {"label": ("JOINT datatype-protection result" if any("+" in o for o in opts) else "DATATYPE-ONLY result (no protected layer in the frozen assignment)")
                              + ": frozen from MEASURED end-to-end validation results under the declared constraints, BEFORE the test set was read.",
                    "assignment_id": aid, "modes_found": asg[aid]["modes"],
                    "selected": [{"layer_id": l, "layer_name": p.layers[l], "num_weights": int(nw[l]), "datatype": S6.parse_option(opts[l])[0], "protection": S6.parse_option(opts[l])[1],
                                  "Q": float(p.Q[l, choice_of[aid][l]]), "R_exposure_pp": float(p.R[l, choice_of[aid][l]]), "R_scale_word_pp": float(p.Rscale[l, choice_of[aid][l]]),
                                  "S_bytes": float(p.S[l, choice_of[aid][l]])} for l in range(L)],
                    "constraints": {"B_avg_bits_per_weight": fin["avg_bits"], "B_bytes": fin["avg_bits"] * N / 8.0, "B_scope": declared["B_scope"], "delta_clean_pp": fin["delta_clean"],
                                    "tau_fault_weights_k8_mean_drop_pp": fin["tau_fault"], "tau_scale_scales_k1_mean_drop_pp": fin["tau_scale"], "A_FP32_validation": fp32["val_accuracy"],
                                    "checked_on": "measured end-to-end validation metrics (not layer-additive proxies)", "declared_constants": declared},
                    "objective": {"formula": "sum x[l,d,p] (Q_norm + lambda R_norm)", "lambda": args.lam, "normalization": args.norm, "value": b["objective_value"]},
                    "solver": "scipy.optimize.milp (HiGHS), exact; candidate assignments from normalization_lambda_sweep.csv; final choice among them by measured feasibility",
                    "storage": b["storage"], "measured_validation_screen": {"clean_validation": b["clean_validation"], "fault_end_to_end": b["fault_end_to_end"]},
                    "measured_validation_final_trials": {"trials_per_distribution": args.final_trials, "sample_count": subset["images"], "fault_end_to_end": final_v},
                    "checkpoint_sha256": ck_sha, "stage2_frozen_config_sha256": frozen, "source_commit": git_info(), "test_set": "evaluated after this file was written; see final_test_evaluation.json"}
        dump_json(sel_json, odir / "assignment.json")                                 # frozen BEFORE the test data is touched
        print(f"FROZEN: {aid} ({b['clean_validation']['accuracy']:.2f} % clean validation, {b['storage']['avg_bits_per_weight']:.3f} bits/weight); evaluating the test set once")
        norm_t = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
        Xt, yt = load_tensors(fdata.get_test_loader(cfg.data_path(), norm_t, cfg.eval_batch_size, cfg.num_workers, subset=args.test_images), device)
        fp32_test = S3.evaluate(model.to(device), Xt, yt)
        clean_t = S3.evaluate(inj.model, Xt, yt)
        ctx_t = {"experiment_id": EXPERIMENT_ID + "-test", "run_id": run_id, "model": MODEL, "dataset": f"CIFAR-10 official test set, {int(yt.numel())} images",
                 "checkpoint_hash": ck_sha, "codec_version": f"mixed|stage2-frozen:{frozen[:12]}|{S6.SCHEMA}"}
        trials_t = {d: [S6.run_protected_trial(inj, pf, Xt, yt, clean_t, ctx_t, assignment_id=aid, dist=d, trial=t, base_seed=args.eval_seed) for t in range(args.test_fault_trials)]
                    for d in dists if d != "stored_k8"}
        test_eval = {"assignment_id": aid, "test_samples": int(yt.numel()), "fp32_test_accuracy": fp32_test["accuracy"], "fp32_test_loss": fp32_test["loss"],
                     "assembled_test_accuracy": clean_t["accuracy"], "assembled_test_loss": clean_t["loss"], "test_clean_drop_pp_vs_fp32": fp32_test["accuracy"] - clean_t["accuracy"],
                     "test_fault": {d: E2.summarize_trials(r) for d, r in trials_t.items()},
                     "note": "official test set, read once, after the assignment was frozen from validation data",
                     "all_restored_exactly": all(r["restored_exactly"] for rs in trials_t.values() for r in rs)}
        dump_json(test_eval, odir / "final_test_evaluation.json")
        sel_json["test_set"] = {"file": "final_test_evaluation.json", "assembled_test_accuracy": clean_t["accuracy"], "fp32_test_accuracy": fp32_test["accuracy"]}
        dump_json(sel_json, odir / "assignment.json")
        print(f"FINAL: test accuracy {clean_t['accuracy']:.2f} % (FP32 {fp32_test['accuracy']:.2f} %) -> assignment.json, final_test_evaluation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
