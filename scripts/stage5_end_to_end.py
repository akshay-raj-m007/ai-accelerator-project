"""J-FAMP Stage 5 end-to-end: assemble the optimiser's assignments into complete mixed-datatype models and MEASURE them (GPU machine).

Runbook 7.2: "the selected assignment must be assembled into a full model and measured end-to-end ... if predicted and measured metrics differ, update
layer-level estimates and repeat refinement." This script does, for every distinct optimal assignment found by stage5_optimize.py (plus the five uniform
single-datatype models as references):
  A. clean validation accuracy / loss of the assembled model (all 5,000 validation images) and actual stored bytes        -> A_clean(x), S(x)
  B. a shortlist (non-dominated in storage, predicted risk and measured clean drop, plus the uniform references) gets end-to-end fault trials on the
     assembled model's stored bytes (declared distributions in stage5_e2e.py), 1,000 validation images, N trials     -> R_fault(x)
  C. predicted-vs-measured comparison and the constraint grid (delta_clean, tau_fault, B): the best MEASURED-feasible assignment per setting.
  D. with --finalize: freeze one assignment under given constraints, write assignment.json and evaluate ONLY that assignment on the official test set.

DATATYPE-ONLY (protection p = none). Validation data only, except --finalize. Resumable (unique trial ids).

  python -u scripts/stage5_end_to_end.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
"""
import argparse
import csv
import datetime as dt
import hashlib
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
from fr import stage2 as S2
from fr import stage3 as S3
from fr import stage4 as S4
from fr import stage5 as S5
from fr import stage5_e2e as E2
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.layer_scores import spearman
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, env_info, git_info, resolve_device
from stage5_optimize import SHORT, load_problem

LONG = {v: k for k, v in SHORT.items()}
EXPERIMENT_ID = "jfamp-resnet18-cifar10-stage5-end-to-end"


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def objective_of(p, choice, lam, norm, risk):
    Rv = p.R if risk == "weights" else p.R + p.Rscale
    qn = S5.normalize(p.Q.reshape(-1), norm).reshape(p.Q.shape)
    rn = S5.normalize(Rv.reshape(-1), norm).reshape(p.Q.shape)
    idx = np.arange(len(p.layers))
    return float((qn + lam * rn)[idx, choice].sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--opt-dir", default=None, help="folder with jfamp_candidate_table.csv, distinct_assignments.csv (default results/<run>/jfamp/optimization)")
    ap.add_argument("--n-fault-images", type=int, default=1000)
    ap.add_argument("--clean-images", type=int, default=None, help="validation images for the clean evaluation (default: all; only for dry runs)")
    ap.add_argument("--trials", type=int, default=50, help="independent fault trials per assignment and distribution")
    ap.add_argument("--max-fault-assignments", type=int, default=25, help="size of the shortlist that gets fault trials (uniform references are extra)")
    ap.add_argument("--include", nargs="*", default=[], help="extra assignment ids that must get fault trials")
    ap.add_argument("--lambda-", dest="lam", type=float, default=1.0, help="lambda of the objective used to rank measured-feasible assignments")
    ap.add_argument("--norm", default="minmax")
    ap.add_argument("--risk", default="weights")
    ap.add_argument("--delta-clean", type=float, nargs="+", default=[0.25, 0.5, 1.0], help="grid, percentage points below A_FP32")
    ap.add_argument("--fault-dist", choices=list(E2.DISTRIBUTIONS), default="weights_k8", help="end-to-end distribution that defines R_fault(x) in the constraint check (scales_k1 counts 0 for assignments without INT layers)")
    ap.add_argument("--tau-fault", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 5.0], help="grid, measured mean accuracy drop (pp) under weights_k8")
    ap.add_argument("--avg-bits", type=float, nargs="+", default=[5, 6, 8, 10, 12, 16], help="grid of storage budgets B (average bits per quantized weight)")
    ap.add_argument("--finalize", action="store_true", help="freeze one assignment under --final-* constraints, write assignment.json, evaluate it on the TEST set")
    ap.add_argument("--final-delta-clean", type=float, default=None)
    ap.add_argument("--final-tau-fault", type=float, default=None)
    ap.add_argument("--final-avg-bits", type=float, default=None)
    ap.add_argument("--test-images", type=int, default=None, help="test images for --finalize (default: all 10,000; only for dry runs)")
    ap.add_argument("--summarize-only", action="store_true", help="no model runs: rebuild summaries from the saved CSV/JSONL")
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    odir = Path(args.opt_dir) if args.opt_dir else base / "optimization"
    odir.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(odir / "candidate_table_meta.json"))
    const_bytes = float(meta["fixed_non_quantized_bytes"])
    p, nw, _ = load_problem(odir / "jfamp_candidate_table.csv")
    L = len(p.layers)
    N = float(nw.sum())
    sweep_rows = list(csv.DictReader(open(odir / "normalization_lambda_sweep.csv")))
    distinct = list(csv.DictReader(open(odir / "distinct_assignments.csv")))
    asg = OrderedDict()
    for r in distinct:
        asg[r["assignment_id"]] = {"formats": [LONG[s] for s in r["assignment"].split(";")], "n_settings": int(r["n_settings"])}
    for f in S5.FORMATS:                                                  # uniform single-datatype references (same assembly code path)
        asg[f"U_{S5.LABEL[f].split()[0].replace('-', '')}"] = {"formats": [f] * L, "n_settings": 0}
    for a in asg.values():
        assert len(a["formats"]) == L and all(f in p.formats for f in a["formats"])
    choice_of = {k: [p.formats.index(f) for f in v["formats"]] for k, v in asg.items()}
    pred = {k: p.totals(c) for k, c in choice_of.items()}

    man2 = json.load(open(qdir / "quantizer_manifest.json"))
    ck = Path(args.checkpoint)
    ck_sha = sha256_file(ck)
    if man2["checkpoint"]["sha256"] != ck_sha:
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    clean_path, trials_path = odir / "end_to_end_clean.csv", odir / "end_to_end_fault_trials.jsonl"
    manifest_path = odir / "end_to_end_manifest.json"
    prev = json.load(open(manifest_path)) if manifest_path.exists() else {}
    run_id = prev.get("run_id") or (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])

    clean_rows = {r["assignment_id"]: r for r in csv.DictReader(open(clean_path))} if clean_path.exists() else {}
    fp32 = prev.get("fp32_reference")
    shortlist = prev.get("shortlist", [])
    t0 = time.time()

    if not args.summarize_only:
        model, ckpt = load_model(ck)
        model.eval()
        mods = quantizable_modules(model)
        lnames = [f"{n}.weight" for n, _ in mods]
        if [n for n, _ in mods] != p.layers:
            sys.exit("layer order differs from the candidate table")
        norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
        val_size, split_seed = split_params(ckpt, cfg)
        loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=args.clean_images)
        Xv, yv = load_tensors(loader, device)
        Xf, yf = Xv[:args.n_fault_images], yv[:args.n_fault_images]
        print(f"run {run_id}: {int(yv.numel())} validation images (clean), first {int(yf.numel())} for fault trials, device {device}")
        codecs = S2.make_codecs(True)
        stores = {}
        for f in S5.FORMATS:
            enc = S2.load_encoded(qdir / man2["formats"][f]["encoded_file"], lnames)
            if S2.content_sha256(enc) != man2["formats"][f]["content_sha256"]:
                sys.exit(f"{f}: encoded weights differ from the frozen content hash")
            stores[f] = S3.EncodedStore(f, enc)
        frozen = man2["frozen_config_sha256"]
        ctx = {"experiment_id": EXPERIMENT_ID, "run_id": run_id, "model": "resnet18_cifar (CIFAR-style: 3x3 stem, no max-pool, 10 classes)",
               "dataset": f"CIFAR-10 validation holdout, first {int(yf.numel())} images", "checkpoint_hash": ck_sha,
               "codec_version": f"mixed|stage2-frozen:{frozen[:12]}|{S3.INJECTOR_VERSION}|{E2.SCHEMA_VERSION}"}

        # ---- FP32 reference on the same validation images
        if fp32 is None:
            ev = S3.evaluate(model.to(device), Xv, yv)
            fp32 = {"val_accuracy": ev["accuracy"], "val_loss": ev["loss"], "val_samples": int(yv.numel())}
            print(f"FP32 reference: {fp32['val_accuracy']:.2f} % / loss {fp32['val_loss']:.4f} on {fp32['val_samples']} validation images")

        def assemble(aid):
            ms = E2.MixedStore(asg[aid]["formats"], stores)
            return ms, E2.MixedFaultInjector(model, ms, codecs, device)

        # ---- A. clean measurements of every assignment
        n_new = 0
        for aid in asg:
            if aid in clean_rows:
                continue
            ms, inj = assemble(aid)
            ev = S3.evaluate(inj.model, Xv, yv)
            sb = ms.stored_bytes()
            pred_bytes = pred[aid]["storage_bytes"]
            assert abs(sb["total_bytes"] - pred_bytes) < 1e-6, f"{aid}: stored bytes {sb['total_bytes']} != table {pred_bytes}"
            c = {f: asg[aid]["formats"].count(f) for f in S5.FORMATS}
            clean_rows[aid] = {"assignment_id": aid, "n_settings": asg[aid]["n_settings"], "val_samples": ev["total"], "clean_val_accuracy": ev["accuracy"],
                               "clean_val_loss": ev["loss"], "clean_drop_pp_vs_fp32": fp32["val_accuracy"] - ev["accuracy"],
                               "delta_ce_vs_fp32": ev["loss"] - fp32["val_loss"], "payload_bytes": sb["payload_bytes"], "scale_bytes": sb["scale_bytes"],
                               "storage_bytes": sb["total_bytes"], "model_storage_bytes_incl_fixed": sb["total_bytes"] + const_bytes,
                               "avg_bits_per_weight": 8 * sb["total_bytes"] / N, "pred_sum_Q": pred[aid]["sum_Q"], "pred_dCE": pred[aid]["predicted_dCE"],
                               "pred_mean_layer_risk_pp": pred[aid]["mean_R_pp"], "pred_mean_layer_risk_plus_scale_pp": pred[aid]["mean_R_plus_scale_pp"],
                               **{f"n_{SHORT[f].lower()}": c[f] for f in S5.FORMATS}, "encoded_hash": ms.hash(),
                               "assignment": ";".join(SHORT[f] for f in asg[aid]["formats"])}
            n_new += 1
            del inj
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if n_new % 20 == 0:
                write_csv(clean_path, list(clean_rows.values()))
                print(f"  clean: {len(clean_rows)}/{len(asg)} assignments")
        write_csv(clean_path, [clean_rows[a] for a in asg])
        print(f"A. clean measurements: {len(asg)} assignments ({n_new} new)")

        # ---- B. shortlist for fault trials: one non-dominated front per risk definition (weights-only R, and R + INT scale-word risk), then the uniform references
        if not shortlist:
            ids = [a for a in asg if not a.startswith("U_")]

            def front_for(risk_col):
                pts = [[float(clean_rows[a]["storage_bytes"]), float(clean_rows[a][risk_col]), float(clean_rows[a]["clean_drop_pp_vs_fp32"])] for a in ids]
                fr = [ids[i] for i in E2.pareto_front(pts)]
                fr.sort(key=lambda a: float(clean_rows[a]["storage_bytes"]))
                half = max(1, args.max_fault_assignments // 2)
                if len(fr) > half:
                    sel = np.unique(np.round(np.linspace(0, len(fr) - 1, half)).astype(int))
                    fr = [fr[i] for i in sel]
                return fr

            front = front_for("pred_mean_layer_risk_pp")
            front += [a for a in front_for("pred_mean_layer_risk_plus_scale_pp") if a not in front]
            shortlist = front + [a for a in args.include if a in asg and a not in front] + [a for a in asg if a.startswith("U_")]
        print(f"B. fault shortlist ({len(shortlist)}): {shortlist}")

        # ---- C. end-to-end fault trials
        log = S4.TrialLog(trials_path)
        if log.duplicates:
            sys.exit(f"duplicate trial ids in {trials_path}")
        n_trials_new = 0
        for aid in shortlist:
            dists = [d for d in E2.DISTRIBUTIONS if d != "scales_k1" or E2.has_scale_words(asg[aid]["formats"])]
            need = [(d, t) for d in dists for t in range(args.trials) if not log.has(f"{aid}|{d}|k{8 if d == 'weights_k8' else 1}|t{t}")]
            if not need:
                continue
            ms, inj = assemble(aid)
            clean = S3.evaluate(inj.model, Xf, yf)
            t1 = time.time()
            for d, t in need:
                rec = E2.run_mixed_trial(inj, Xf, yf, clean, ctx, assignment_id=aid, dist=d, k=8 if d == "weights_k8" else 1, trial=t, base_seed=cfg.seed)
                if not rec["restored_exactly"] or not rec["flip_applied_to_encoded_bytes"]:
                    sys.exit(f"{aid}: restoration / flip check failed")
                log.append(rec)
                n_trials_new += 1
            print(f"  {aid}: {len(need)} trials in {time.time() - t1:.0f} s (clean {clean['accuracy']:.2f} % on {int(yf.numel())} images)")
            del inj
            if device.type == "cuda":
                torch.cuda.empty_cache()
        log.close()
        print(f"C. fault trials: {n_trials_new} new")
        dump_json({"run_id": run_id, "fp32_reference": fp32, "shortlist": shortlist, "checkpoint_sha256": ck_sha, "stage2_frozen_config_sha256": frozen,
                   "settings": {"trials": args.trials, "n_fault_images": args.n_fault_images, "base_seed": cfg.seed, "distributions": E2.DISTRIBUTIONS},
                   "git": git_info(), "environment": env_info(), "seconds_last_session": time.time() - t0}, manifest_path)
        prev = json.load(open(manifest_path))

    # ======================================================================== summaries (from saved CSV / JSONL only)
    clean_rows = {r["assignment_id"]: r for r in csv.DictReader(open(clean_path))}
    fp32 = prev["fp32_reference"]
    shortlist = prev["shortlist"]
    recs = S4.load_records(trials_path) if trials_path.exists() else []
    by = {}
    for r in recs:
        by.setdefault((r["assignment_id"], r["distribution"]), []).append(r)
    trials_expected = prev["settings"]["trials"]
    problems = []
    for (aid, d), rs in by.items():
        if len(rs) != trials_expected or len({r["trial_id"] for r in rs}) != len(rs):
            problems.append(f"{aid}/{d}: {len(rs)} trials")
    items = []
    for aid in asg:
        cr = clean_rows[aid]
        item = {"assignment_id": aid, "datatype_per_layer": dict(zip(p.layers, asg[aid]["formats"])), "protection": "none",
                "counts": {f: asg[aid]["formats"].count(f) for f in S5.FORMATS},
                "storage": {"actual_stored_bytes": int(cr["storage_bytes"]), "predicted_from_table_bytes": pred[aid]["storage_bytes"],
                            "avg_bits_per_weight": float(cr["avg_bits_per_weight"]), "model_bytes_incl_fixed": float(cr["model_storage_bytes_incl_fixed"])},
                "clean_validation": {"samples": int(cr["val_samples"]), "accuracy": float(cr["clean_val_accuracy"]), "drop_pp_vs_fp32": float(cr["clean_drop_pp_vs_fp32"]),
                                     "loss": float(cr["clean_val_loss"]), "delta_ce_vs_fp32": float(cr["delta_ce_vs_fp32"])},
                "predicted_layer_additive": {"sum_Q": pred[aid]["sum_Q"], "dCE": pred[aid]["predicted_dCE"], "mean_layer_risk_pp": pred[aid]["mean_R_pp"],
                                             "mean_layer_risk_plus_scale_pp": pred[aid]["mean_R_plus_scale_pp"]},
                "objective_value": objective_of(p, choice_of[aid], args.lam, args.norm, args.risk),
                "fault_end_to_end": {d: E2.summarize_trials(by[(aid, d)]) for d in E2.DISTRIBUTIONS if (aid, d) in by}}
        items.append(item)
    flat = []
    for it in items:
        row = {"assignment_id": it["assignment_id"], "avg_bits": it["storage"]["avg_bits_per_weight"], "storage_bytes": it["storage"]["actual_stored_bytes"],
               "clean_val_accuracy": it["clean_validation"]["accuracy"], "clean_drop_pp": it["clean_validation"]["drop_pp_vs_fp32"],
               "delta_ce": it["clean_validation"]["delta_ce_vs_fp32"], "pred_dCE": it["predicted_layer_additive"]["dCE"],
               "pred_mean_layer_risk_pp": it["predicted_layer_additive"]["mean_layer_risk_pp"], "objective": it["objective_value"]}
        for d in E2.DISTRIBUTIONS:
            s = it["fault_end_to_end"].get(d)
            for key in ("n_trials", "sample_count", "drop_pp_mean", "drop_pp_median", "drop_pp_p90", "drop_pp_worst", "failure_rate"):
                row[f"{d}_{key}"] = "" if not s else s[key]
        row["assignment"] = ";".join(SHORT[f] for f in asg[it["assignment_id"]]["formats"])
        flat.append(row)
    write_csv(odir / "end_to_end_assignments.csv", flat)

    # ---- predicted vs measured
    meas = [it for it in items if "weights_k8" in it["fault_end_to_end"]]       # assignments with end-to-end fault trials
    comp = {"note": "layer-additive estimates vs end-to-end measurements of the assembled models", "n_assignments_clean": len(items), "n_assignments_fault": len(meas)}
    if len(items) > 2:
        comp["spearman_predicted_dCE_vs_measured_dCE"] = spearman([i["predicted_layer_additive"]["dCE"] for i in items], [i["clean_validation"]["delta_ce_vs_fp32"] for i in items])
        comp["spearman_predicted_Q_vs_measured_clean_drop_pp"] = spearman([i["predicted_layer_additive"]["sum_Q"] for i in items], [i["clean_validation"]["drop_pp_vs_fp32"] for i in items])
    if len(meas) > 2:
        comp["spearman_predicted_mean_layer_risk_vs_measured_weights_k8_drop"] = spearman(
            [i["predicted_layer_additive"]["mean_layer_risk_pp"] for i in meas], [i["fault_end_to_end"]["weights_k8"]["drop_pp_mean"] for i in meas])
        comp["per_assignment"] = [{"assignment_id": i["assignment_id"], "predicted_dCE": i["predicted_layer_additive"]["dCE"], "measured_dCE": i["clean_validation"]["delta_ce_vs_fp32"],
                                   "predicted_mean_layer_risk_pp": i["predicted_layer_additive"]["mean_layer_risk_pp"],
                                   "measured_weights_k8_drop_pp_mean": i["fault_end_to_end"]["weights_k8"]["drop_pp_mean"],
                                   "measured_weights_k8_drop_pp_p90": i["fault_end_to_end"]["weights_k8"]["drop_pp_p90"]} for i in meas]

    # ---- constraint grid: best MEASURED-feasible assignment per (delta_clean, tau_fault, B)
    def rfault(it):
        s = it["fault_end_to_end"].get(args.fault_dist)
        return float(s["drop_pp_mean"]) if s else 0.0          # scales_k1 is undefined without INT layers: no scale words, no exposure

    def feasible(delta, tau, bits):
        out = []
        for it in meas:
            ok = (it["clean_validation"]["drop_pp_vs_fp32"] <= delta + 1e-9 and rfault(it) <= tau + 1e-9
                  and it["storage"]["avg_bits_per_weight"] <= bits + 1e-9)
            if ok:
                out.append(it)
        return out

    def best(delta, tau, bits):
        f = feasible(delta, tau, bits)
        return min(f, key=lambda it: (it["objective_value"], it["storage"]["actual_stored_bytes"])) if f else None

    grid = []
    for dlt in args.delta_clean:
        for tau in args.tau_fault:
            for bits in args.avg_bits:
                b = best(dlt, tau, bits)
                grid.append({"delta_clean_pp": dlt, "tau_fault_pp": tau, "B_avg_bits": bits, "n_feasible": len(feasible(dlt, tau, bits)),
                             "selected_assignment": b["assignment_id"] if b else "none feasible",
                             "objective": "" if not b else b["objective_value"], "avg_bits": "" if not b else b["storage"]["avg_bits_per_weight"],
                             "clean_drop_pp": "" if not b else b["clean_validation"]["drop_pp_vs_fp32"],
                             "R_fault_pp_mean_" + args.fault_dist: "" if not b else rfault(b)})
    write_csv(odir / "constraint_grid_selection.csv", grid)
    dump_json({"label": "DATATYPE-ONLY (protection p = none). Measured end to end on assembled models; validation data only unless final_test_evaluation.json exists.",
               "fp32_reference": fp32, "protocol": {"clean": "all 5,000 validation images", "fault": f"first {prev['settings']['n_fault_images']} validation images, {trials_expected} trials per distribution",
                                                    "distributions": "weights_k8: 8 distinct weights over the whole assembled model, bit uniform over the weight's own encoding; scales_k1: 1 INT scale word, 1 of 32 bits (extension)"},
               "fault_shortlist_rule": "union of two fronts among the optimiser's distinct assignments, each non-dominated in (actual storage, predicted mean layer risk, measured clean drop) - one with weights-only risk, one with risk plus INT scale-word risk - each evenly thinned to half the size limit, plus the five uniform references",
               "shortlist": shortlist, "incomplete_cells": problems, "predicted_vs_measured": comp, "assignments": items}, odir / "end_to_end_assignment_metrics.json")
    print(f"summaries written to {odir}  (incomplete trial cells: {len(problems)})")

    # ======================================================================== finalize: freeze one assignment, test set once
    if args.finalize:
        if None in (args.final_delta_clean, args.final_tau_fault, args.final_avg_bits):
            sys.exit("--finalize needs --final-delta-clean, --final-tau-fault and --final-avg-bits")
        b = best(args.final_delta_clean, args.final_tau_fault, args.final_avg_bits)
        if b is None:
            sys.exit("no measured-feasible assignment under these constraints; see constraint_grid_selection.csv")
        aid = b["assignment_id"]
        sw = [r for r in sweep_rows if r["assignment"] == ";".join(SHORT[f] for f in asg[aid]["formats"]) and r["status"] == "optimal"]
        model, ckpt = load_model(ck)
        model.eval()
        mods = quantizable_modules(model)
        lnames = [f"{n}.weight" for n, _ in mods]
        norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
        tl = fdata.get_test_loader(cfg.data_path(), norm, cfg.eval_batch_size, cfg.num_workers, subset=args.test_images)
        Xt, yt = load_tensors(tl, device)
        codecs = S2.make_codecs(True)
        stores = {}
        for f in S5.FORMATS:
            enc = S2.load_encoded(qdir / man2["formats"][f]["encoded_file"], lnames)
            if S2.content_sha256(enc) != man2["formats"][f]["content_sha256"]:
                sys.exit(f"{f}: encoded weights differ from the frozen content hash")
            stores[f] = S3.EncodedStore(f, enc)
        ms = E2.MixedStore(asg[aid]["formats"], stores)
        inj = E2.MixedFaultInjector(model, ms, codecs, device)
        fp32_test = S3.evaluate(model.to(device), Xt, yt)
        clean_t = S3.evaluate(inj.model, Xt, yt)
        ctx = {"experiment_id": EXPERIMENT_ID + "-test", "run_id": run_id, "model": "resnet18_cifar", "dataset": f"CIFAR-10 official test set, {int(yt.numel())} images",
               "checkpoint_hash": ck_sha, "codec_version": f"mixed|{E2.SCHEMA_VERSION}"}
        trials_t = [E2.run_mixed_trial(inj, Xt, yt, clean_t, ctx, assignment_id=aid, dist="weights_k8", k=8, trial=t, base_seed=cfg.seed) for t in range(args.trials)]
        test_eval = {"assignment_id": aid, "test_samples": int(yt.numel()), "fp32_test_accuracy": fp32_test["accuracy"], "assembled_test_accuracy": clean_t["accuracy"],
                     "test_clean_drop_pp_vs_fp32": fp32_test["accuracy"] - clean_t["accuracy"], "test_weights_k8": E2.summarize_trials(trials_t),
                     "note": "official test set, evaluated once for the frozen assignment; do not compare with the 1,000-image validation fault screen as the same protocol",
                     "all_restored_exactly": all(r["restored_exactly"] for r in trials_t)}
        dump_json(test_eval, odir / "final_test_evaluation.json")
        dump_json({"label": "DATATYPE-ONLY (protection p = none). Frozen from MEASURED end-to-end validation results under the stated constraints.",
                   "selected": [{"layer_id": l, "layer_name": p.layers[l], "num_weights": int(nw[l]), "datatype": asg[aid]["formats"][l], "protection": "none",
                                 "Q": float(p.Q[l, choice_of[aid][l]]), "R_any_k8_mean_drop_pp": float(p.R[l, choice_of[aid][l]]), "R_scale_any_k1_mean_drop_pp": float(p.Rscale[l, choice_of[aid][l]]),
                                 "S_bytes": float(p.S[l, choice_of[aid][l]])} for l in range(L)],
                   "assignment_id": aid,
                   "constraints": {"B_avg_bits_per_weight": args.final_avg_bits, "delta_clean_pp": args.final_delta_clean, "tau_fault_weights_k8_mean_drop_pp": args.final_tau_fault,
                                   "A_FP32_validation": fp32["val_accuracy"], "checked_on": "measured end-to-end validation metrics (not layer-additive proxies)"},
                   "objective": {"formula": "sum x[l,d,p] (Q_norm + lambda R_norm)", "lambda": args.lam, "normalization": args.norm, "risk_variant": args.risk, "value": b["objective_value"]},
                   "optimiser_settings_that_produced_it": sw[:3], "solver": "scipy.optimize.milp (HiGHS), exact; candidate assignments from normalization_lambda_sweep.csv",
                   "measured": {"storage": b["storage"], "clean_validation": b["clean_validation"], "fault_end_to_end": b["fault_end_to_end"]},
                   "test_set": {"file": "final_test_evaluation.json"}}, odir / "assignment.json")
        print(f"FINAL: {aid} frozen; test accuracy {clean_t['accuracy']:.2f} % (FP32 {fp32_test['accuracy']:.2f} %) -> assignment.json, final_test_evaluation.json")


if __name__ == "__main__":
    main()
