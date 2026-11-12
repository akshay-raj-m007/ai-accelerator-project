"""J-FAMP Stage 5 end-to-end: assemble the optimiser's assignments into complete mixed-datatype (and, where chosen, protected) models and MEASURE them (GPU machine).

Runbook 7.2: "the selected assignment must be assembled into a full model and measured end-to-end ... if predicted and measured metrics differ, update layer-level
estimates and repeat refinement." This script does, for every distinct optimal assignment found by stage5_optimize.py (plus uniform single-datatype references, with and
without scale_tmr for INT):
  A. clean validation accuracy / loss of the assembled model (validation images 2,048..4,999, disjoint from the 2,048 Fisher images) and actual stored bytes -> A_clean(x), S(x)
  B. a shortlist (non-dominated fronts of the datatype-only and of the protected assignments, plus the references) gets end-to-end fault trials on the assembled model's
     STORED bytes (declared distributions in src/fr/stage5_e2e.py), 1,000 validation images -> R_fault(x) (primary: weights_k8) and the scale-word metric (scales_k1)
  C. predicted-vs-measured comparison and the constraint grid (delta_clean, tau_fault, B, tau_scale): the best MEASURED-feasible assignment per setting.
  D. with --finalize: freeze one assignment under the agreed constraints, run MORE fault trials on it, write assignment.json and evaluate ONLY that assignment on the
     official test set.

Validation data only, except --finalize. Resumable (unique trial ids).

  python -u scripts/stage5_end_to_end.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
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
from fr import protect as P
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
from stage5_optimize import LONG, SHORT, load_problem

EXPERIMENT_ID = "jfamp-resnet18-cifar10-stage5-end-to-end"
DIST_K = {"weights_k8": 8, "scales_k1": 1}


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def objective_of(p, choice, lam, norm):
    qn = S5.normalize(p.Q.reshape(-1), norm).reshape(p.Q.shape)
    rn = S5.normalize(p.R.reshape(-1), norm).reshape(p.Q.shape)
    idx = np.arange(len(p.layers))
    return float((qn + lam * rn)[idx, choice].sum())


def scale_bits(tok):
    fmt, prot = P.parse_option(tok)
    if S3.SPECS[fmt]["kind"] != "int":
        return 0
    return 32 * (P.COPIES if prot == P.PROTECTION_TMR else 1)


def predicted_scale_drop(p, choice):
    """Expected accuracy drop (pp) of the scales_k1 distribution: one flip uniform over all stored scale bits; per-layer severity = pooled scale-word risk of its option."""
    bits = np.array([scale_bits(p.formats[c]) for c in choice], dtype=float)
    if bits.sum() == 0:
        return 0.0
    return float((bits / bits.sum() * p.Rscale[np.arange(len(choice)), choice]).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--opt-dir", default=None, help="folder with jfamp_candidate_table.csv, distinct_assignments.csv (default results/<run>/jfamp/optimization)")
    ap.add_argument("--n-fault-images", type=int, default=1000)
    ap.add_argument("--clean-images", type=int, default=None, help="validation images for the clean evaluation after the skip (default: all remaining; only for dry runs)")
    ap.add_argument("--clean-skip", type=int, default=2048, help="skip the first N validation images (the exact-Fisher images used for Q) in the clean evaluation")
    ap.add_argument("--q-column", default="Q_fisher_val_exact", help="Q column of the candidate table (must match the one used by stage5_optimize.py)")
    ap.add_argument("--r-column", default="R_exposure_k8_pp", help="R column of the candidate table (must match the one used by stage5_optimize.py)")
    ap.add_argument("--trials", type=int, default=100, help="independent fault trials per assignment and distribution")
    ap.add_argument("--max-fault-assignments", type=int, default=24, help="size of the shortlist that gets fault trials (references are extra)")
    ap.add_argument("--include", nargs="*", default=[], help="extra assignment ids that must get fault trials")
    ap.add_argument("--lambda-", dest="lam", type=float, default=1.0, help="lambda of the objective used to rank measured-feasible assignments")
    ap.add_argument("--norm", default="minmax")
    ap.add_argument("--delta-clean", type=float, nargs="+", default=[0.25, 0.5, 1.0], help="grid, percentage points below A_FP32")
    ap.add_argument("--tau-fault", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0], help="grid, measured mean accuracy drop (pp) under weights_k8 (primary R_fault)")
    ap.add_argument("--tau-scale", type=float, nargs="+", default=[1.0], help="grid, measured mean accuracy drop (pp) under scales_k1; a run without this limit is always added")
    ap.add_argument("--avg-bits", type=float, nargs="+", default=[5, 6, 8, 10, 12, 16], help="grid of storage budgets B (average bits per quantized weight)")
    ap.add_argument("--finalize", action="store_true", help="freeze one assignment under --final-* constraints, write assignment.json, evaluate it on the TEST set")
    ap.add_argument("--final-delta-clean", type=float, default=None)
    ap.add_argument("--final-tau-fault", type=float, default=None)
    ap.add_argument("--final-avg-bits", type=float, default=None)
    ap.add_argument("--final-tau-scale", type=float, default=None, help="measured scales_k1 limit for the final selection (joint analysis); omit for the datatype-only report")
    ap.add_argument("--final-trials", type=int, default=300, help="fault trials per distribution for the frozen assignment on the 1,000 validation fault images")
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
    p, nw, _ = load_problem(odir / "jfamp_candidate_table.csv", args.q_column, args.r_column, "joint")
    L = len(p.layers)
    N = float(nw.sum())
    sweep_rows = list(csv.DictReader(open(odir / "normalization_lambda_sweep.csv")))
    distinct = list(csv.DictReader(open(odir / "distinct_assignments.csv")))
    asg = OrderedDict()
    for r in distinct:
        asg[r["assignment_id"]] = {"options": [LONG[s] for s in r["assignment"].split(";")], "n_settings": int(r["n_settings"]), "modes": r["found_in_modes"]}
    for f in S5.FORMATS:                                                  # uniform single-datatype references (same assembly code path)
        asg[f"U_{S5.LABEL[f].split()[0].replace('-', '')}"] = {"options": [f] * L, "n_settings": 0, "modes": "reference"}
    for f in ("int8_pt", "int4_pt"):
        if f + "+scale_tmr" in p.formats:
            asg[f"U_{S5.LABEL[f].split()[0]}_TMR"] = {"options": [f + "+scale_tmr"] * L, "n_settings": 0, "modes": "reference"}
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
    clean_path, trials_path = odir / "end_to_end_clean.csv", odir / "end_to_end_fault_trials.jsonl"
    manifest_path = odir / "end_to_end_manifest.json"
    prev = json.load(open(manifest_path)) if manifest_path.exists() else {}
    run_id = prev.get("run_id") or (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])

    clean_rows = {r["assignment_id"]: r for r in csv.DictReader(open(clean_path))} if clean_path.exists() else {}
    fp32 = prev.get("fp32_reference")
    shortlist = prev.get("shortlist", [])
    t0 = time.time()

    def load_stores(lnames):
        stores = {}
        for f in S5.FORMATS:
            enc = S2.load_encoded(qdir / man2["formats"][f]["encoded_file"], lnames)
            if S2.content_sha256(enc) != man2["formats"][f]["content_sha256"]:
                sys.exit(f"{f}: encoded weights differ from the frozen content hash")
            stores[f] = S3.EncodedStore(f, enc)
        return stores

    if not args.summarize_only:
        model, ckpt = load_model(ck)
        model.eval()
        mods = quantizable_modules(model)
        lnames = [f"{n}.weight" for n, _ in mods]
        if [n for n, _ in mods] != p.layers:
            sys.exit("layer order differs from the candidate table")
        norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
        val_size, split_seed = split_params(ckpt, cfg)
        loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=args.clean_images, skip=args.clean_skip)
        Xv, yv = load_tensors(loader, device)
        # fault trials use the first images of the validation split, as in Stage 4 (they overlap the Fisher images; the Fisher uses gradients, the fault result a measured accuracy drop)
        Xf, yf = load_tensors(fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=args.n_fault_images), device)
        key = (args.q_column, args.r_column, args.clean_skip)
        if prev and prev.get("settings") and (prev["settings"].get("q_column"), prev["settings"].get("r_column"), prev["settings"].get("clean_skip")) != key:
            sys.exit("end_to_end_manifest.json was made with different --q-column / --r-column / --clean-skip: use another --opt-dir or delete the old end_to_end_* files")
        print(f"run {run_id}: {int(yv.numel())} validation images (clean, skipping the first {args.clean_skip}), first {int(yf.numel())} for fault trials, device {device}")
        codecs = S2.make_codecs(True)
        stores = load_stores(lnames)
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
            ms = E2.MixedStore(asg[aid]["options"], stores)
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
            c = {o: asg[aid]["options"].count(o) for o in p.formats}
            clean_rows[aid] = {"assignment_id": aid, "modes": asg[aid]["modes"], "n_settings": asg[aid]["n_settings"], "val_samples": ev["total"], "clean_val_accuracy": ev["accuracy"],
                               "clean_val_loss": ev["loss"], "clean_drop_pp_vs_fp32": fp32["val_accuracy"] - ev["accuracy"], "delta_ce_vs_fp32": ev["loss"] - fp32["val_loss"],
                               "payload_bytes": sb["payload_bytes"], "scale_bytes": sb["scale_bytes"], "protection_overhead_bytes": sb["protection_overhead_bytes"],
                               "storage_bytes": sb["total_bytes"], "model_storage_bytes_incl_fixed": sb["total_bytes"] + const_bytes, "avg_bits_per_weight": 8 * sb["total_bytes"] / N,
                               "pred_sum_Q": pred[aid]["sum_Q"], "pred_dCE": pred[aid]["predicted_dCE"], "pred_weight_fault_drop_pp": pred[aid]["sum_R_pp"],
                               "pred_scale_word_drop_pp": pred_scale[aid], "pred_max_layer_scale_risk_pp": pred[aid]["max_layer_scale_risk_pp"],
                               **{f"n_{SHORT[o].lower().replace('+t', '_tmr')}": c[o] for o in p.formats}, "encoded_hash": ms.hash(),
                               "assignment": ";".join(SHORT[o] for o in asg[aid]["options"])}
            n_new += 1
            del inj
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if n_new % 20 == 0:
                write_csv(clean_path, list(clean_rows.values()))
                print(f"  clean: {len(clean_rows)}/{len(asg)} assignments")
        write_csv(clean_path, [clean_rows[a] for a in asg])
        print(f"A. clean measurements: {len(asg)} assignments ({n_new} new)")

        # ---- B. shortlist for fault trials: a non-dominated front of the datatype-only assignments and one of the protected assignments, then the references
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

        # ---- C. end-to-end fault trials
        log = S4.TrialLog(trials_path)
        if log.duplicates:
            sys.exit(f"duplicate trial ids in {trials_path}")
        n_trials_new = 0
        for aid in shortlist:
            dists = [d for d in E2.DISTRIBUTIONS if d != "scales_k1" or E2.has_scale_words(asg[aid]["options"])]
            need = [(d, t) for d in dists for t in range(args.trials) if not log.has(f"{aid}|{d}|k{DIST_K[d]}|t{t}")]
            if not need:
                continue
            ms, inj = assemble(aid)
            clean = S3.evaluate(inj.model, Xf, yf)
            t1 = time.time()
            for d, t in need:
                rec = E2.run_mixed_trial(inj, Xf, yf, clean, ctx, assignment_id=aid, dist=d, k=DIST_K[d], trial=t, base_seed=cfg.seed)
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
                   "settings": {"q_column": args.q_column, "r_column": args.r_column, "clean_skip": args.clean_skip, "trials": args.trials, "n_fault_images": args.n_fault_images,
                                "base_seed": cfg.seed, "distributions": E2.DISTRIBUTIONS},
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
        opts = asg[aid]["options"]
        item = {"assignment_id": aid, "modes": asg[aid]["modes"], "datatype_per_layer": dict(zip(p.layers, [P.parse_option(o)[0] for o in opts])),
                "protection_per_layer": dict(zip(p.layers, [P.parse_option(o)[1] for o in opts])), "counts": {o: opts.count(o) for o in p.formats},
                "n_protected_layers": sum(1 for o in opts if "+" in o),
                "storage": {"actual_stored_bytes": int(cr["storage_bytes"]), "predicted_from_table_bytes": pred[aid]["storage_bytes"], "protection_overhead_bytes": int(cr["protection_overhead_bytes"]),
                            "avg_bits_per_weight": float(cr["avg_bits_per_weight"]), "model_bytes_incl_fixed": float(cr["model_storage_bytes_incl_fixed"])},
                "clean_validation": {"samples": int(cr["val_samples"]), "accuracy": float(cr["clean_val_accuracy"]), "drop_pp_vs_fp32": float(cr["clean_drop_pp_vs_fp32"]),
                                     "loss": float(cr["clean_val_loss"]), "delta_ce_vs_fp32": float(cr["delta_ce_vs_fp32"])},
                "predicted_layer_additive": {"sum_Q": pred[aid]["sum_Q"], "dCE": pred[aid]["predicted_dCE"], "weight_fault_drop_pp": pred[aid]["sum_R_pp"],
                                             "scale_word_drop_pp": pred_scale[aid], "max_layer_scale_risk_pp": pred[aid]["max_layer_scale_risk_pp"]},
                "objective_value": objective_of(p, choice_of[aid], args.lam, args.norm),
                "fault_end_to_end": {d: E2.summarize_trials(by[(aid, d)]) for d in E2.DISTRIBUTIONS if (aid, d) in by}}
        items.append(item)
    flat = []
    for it in items:
        row = {"assignment_id": it["assignment_id"], "modes": it["modes"], "n_protected_layers": it["n_protected_layers"], "avg_bits": it["storage"]["avg_bits_per_weight"],
               "storage_bytes": it["storage"]["actual_stored_bytes"], "clean_val_accuracy": it["clean_validation"]["accuracy"], "clean_drop_pp": it["clean_validation"]["drop_pp_vs_fp32"],
               "delta_ce": it["clean_validation"]["delta_ce_vs_fp32"], "pred_dCE": it["predicted_layer_additive"]["dCE"],
               "pred_weight_fault_drop_pp": it["predicted_layer_additive"]["weight_fault_drop_pp"], "pred_scale_word_drop_pp": it["predicted_layer_additive"]["scale_word_drop_pp"],
               "objective": it["objective_value"]}
        for d in E2.DISTRIBUTIONS:
            s = it["fault_end_to_end"].get(d)
            for k_ in ("n_trials", "sample_count", "drop_pp_mean", "drop_pp_median", "drop_pp_p90", "drop_pp_worst", "failure_rate"):
                row[f"{d}_{k_}"] = "" if not s else s[k_]
        row["assignment"] = ";".join(SHORT[o] for o in asg[it["assignment_id"]]["options"])
        flat.append(row)
    write_csv(odir / "end_to_end_assignments.csv", flat)

    # ---- predicted vs measured
    meas = [it for it in items if "weights_k8" in it["fault_end_to_end"]]       # assignments with end-to-end fault trials
    comp = {"note": "layer-additive estimates vs end-to-end measurements of the assembled models", "n_assignments_clean": len(items), "n_assignments_fault": len(meas)}
    if len(items) > 2:
        comp["spearman_predicted_dCE_vs_measured_dCE"] = spearman([i["predicted_layer_additive"]["dCE"] for i in items], [i["clean_validation"]["delta_ce_vs_fp32"] for i in items])
        comp["spearman_predicted_Q_vs_measured_clean_drop_pp"] = spearman([i["predicted_layer_additive"]["sum_Q"] for i in items], [i["clean_validation"]["drop_pp_vs_fp32"] for i in items])
    if len(meas) > 2:
        pw = [i["predicted_layer_additive"]["weight_fault_drop_pp"] for i in meas]
        mw = [i["fault_end_to_end"]["weights_k8"]["drop_pp_mean"] for i in meas]
        comp["spearman_predicted_weight_fault_drop_vs_measured_weights_k8"] = spearman(pw, mw)
        comp["weights_k8_mean_predicted_pp"], comp["weights_k8_mean_measured_pp"] = float(np.mean(pw)), float(np.mean(mw))
        sm = [i for i in meas if "scales_k1" in i["fault_end_to_end"]]
        if len(sm) > 2:
            comp["spearman_predicted_scale_drop_vs_measured_scales_k1"] = spearman([i["predicted_layer_additive"]["scale_word_drop_pp"] for i in sm],
                                                                                   [i["fault_end_to_end"]["scales_k1"]["drop_pp_mean"] for i in sm])
        comp["per_assignment"] = [{"assignment_id": i["assignment_id"], "predicted_dCE": i["predicted_layer_additive"]["dCE"], "measured_dCE": i["clean_validation"]["delta_ce_vs_fp32"],
                                   "predicted_weight_fault_drop_pp": i["predicted_layer_additive"]["weight_fault_drop_pp"],
                                   "measured_weights_k8_drop_pp_mean": i["fault_end_to_end"]["weights_k8"]["drop_pp_mean"],
                                   "measured_weights_k8_drop_pp_p90": i["fault_end_to_end"]["weights_k8"]["drop_pp_p90"],
                                   "predicted_scale_word_drop_pp": i["predicted_layer_additive"]["scale_word_drop_pp"],
                                   "measured_scales_k1_drop_pp_mean": i["fault_end_to_end"].get("scales_k1", {}).get("drop_pp_mean")} for i in meas]

    # ---- constraint grid: best MEASURED-feasible assignment per (delta_clean, tau_fault, B, tau_scale)
    def r_weights(it):
        return float(it["fault_end_to_end"]["weights_k8"]["drop_pp_mean"])

    def r_scale(it):
        s = it["fault_end_to_end"].get("scales_k1")
        return float(s["drop_pp_mean"]) if s else 0.0          # no INT layer -> no scale words -> no exposure

    def feasible(delta, tau, bits, tau_s):
        return [it for it in meas if it["clean_validation"]["drop_pp_vs_fp32"] <= delta + 1e-9 and r_weights(it) <= tau + 1e-9
                and it["storage"]["avg_bits_per_weight"] <= bits + 1e-9 and (tau_s is None or r_scale(it) <= tau_s + 1e-9)]

    def best(delta, tau, bits, tau_s):
        f = feasible(delta, tau, bits, tau_s)
        return min(f, key=lambda it: (it["objective_value"], it["storage"]["actual_stored_bytes"])) if f else None

    grid = []
    for tau_s in [None] + list(args.tau_scale):
        for dlt in args.delta_clean:
            for tau in args.tau_fault:
                for bits in args.avg_bits:
                    b = best(dlt, tau, bits, tau_s)
                    grid.append({"tau_scale_pp": "" if tau_s is None else tau_s, "delta_clean_pp": dlt, "tau_fault_pp": tau, "B_avg_bits": bits, "n_feasible": len(feasible(dlt, tau, bits, tau_s)),
                                 "selected_assignment": b["assignment_id"] if b else "none feasible", "objective": "" if not b else b["objective_value"],
                                 "avg_bits": "" if not b else b["storage"]["avg_bits_per_weight"], "clean_drop_pp": "" if not b else b["clean_validation"]["drop_pp_vs_fp32"],
                                 "R_fault_weights_k8_pp": "" if not b else r_weights(b), "scales_k1_pp": "" if not b else r_scale(b), "n_protected_layers": "" if not b else b["n_protected_layers"]})
    write_csv(odir / "constraint_grid_selection.csv", grid)
    dump_json({"label": "Measured end to end on assembled models (validation data only unless final_test_evaluation.json exists). Protection options in the table: none and scale_tmr (both implemented and measured).",
               "fp32_reference": fp32, "protocol": {"clean": f"validation images {prev['settings']['clean_skip']}..4999 (disjoint from the Fisher images 0..2047)", "q_column": prev["settings"]["q_column"],
                                                    "r_column": prev["settings"]["r_column"], "fault": f"first {prev['settings']['n_fault_images']} validation images, {trials_expected} trials per distribution",
                                                    "distributions": "weights_k8 (PRIMARY): 8 distinct weights over the whole assembled model, bit uniform over the weight's own encoding; "
                                                                     "scales_k1 (SECOND metric): 1 flip uniform over all stored scale bits (32 per unprotected INT layer, 3 x 32 per scale_tmr layer)"},
               "fault_shortlist_rule": "non-dominated front (storage, predicted weight-fault drop, largest scale-word risk, measured clean drop) of the datatype-only assignments and of the protected assignments, each evenly thinned to half the size limit, plus the references",
               "shortlist": shortlist, "incomplete_cells": problems, "predicted_vs_measured": comp, "assignments": items}, odir / "end_to_end_assignment_metrics.json")
    print(f"summaries written to {odir}  (incomplete trial cells: {len(problems)})")

    # ======================================================================== finalize: freeze one assignment, more trials, test set once
    if args.finalize:
        if None in (args.final_delta_clean, args.final_tau_fault, args.final_avg_bits):
            sys.exit("--finalize needs --final-delta-clean, --final-tau-fault and --final-avg-bits")
        b = best(args.final_delta_clean, args.final_tau_fault, args.final_avg_bits, args.final_tau_scale)
        if b is None:
            sys.exit("no measured-feasible assignment under these constraints; see constraint_grid_selection.csv")
        aid = b["assignment_id"]
        opts = asg[aid]["options"]
        sw = [r for r in sweep_rows if r["assignment"] == ";".join(SHORT[o] for o in opts) and r["status"] == "optimal"]
        model, ckpt = load_model(ck)
        model.eval()
        mods = quantizable_modules(model)
        lnames = [f"{n}.weight" for n, _ in mods]
        norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
        val_size, split_seed = split_params(ckpt, cfg)
        Xf, yf = load_tensors(fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=prev["settings"]["n_fault_images"]), device)
        Xt, yt = load_tensors(fdata.get_test_loader(cfg.data_path(), norm, cfg.eval_batch_size, cfg.num_workers, subset=args.test_images), device)
        codecs = S2.make_codecs(True)
        stores = load_stores(lnames)
        ms = E2.MixedStore(opts, stores)
        inj = E2.MixedFaultInjector(model, ms, codecs, device)
        dists = [d for d in E2.DISTRIBUTIONS if d != "scales_k1" or E2.has_scale_words(opts)]
        # more validation fault trials on the frozen assignment (a separate id space: FINAL)
        ctx_v = {"experiment_id": EXPERIMENT_ID + "-final-validation", "run_id": run_id, "model": "resnet18_cifar", "dataset": f"CIFAR-10 validation holdout, first {int(yf.numel())} images",
                 "checkpoint_hash": ck_sha, "codec_version": f"mixed|{E2.SCHEMA_VERSION}"}
        clean_f = S3.evaluate(inj.model, Xf, yf)
        final_v = {}
        with open(odir / "final_validation_fault_trials.jsonl", "w", encoding="utf8") as fh:
            for d in dists:
                rs = [E2.run_mixed_trial(inj, Xf, yf, clean_f, ctx_v, assignment_id=f"FINAL_{aid}", dist=d, k=DIST_K[d], trial=t, base_seed=cfg.seed) for t in range(args.final_trials)]
                for r in rs:
                    fh.write(json.dumps(r) + "\n")
                final_v[d] = E2.summarize_trials(rs)
        fp32_test = S3.evaluate(model.to(device), Xt, yt)
        clean_t = S3.evaluate(inj.model, Xt, yt)
        ctx = {"experiment_id": EXPERIMENT_ID + "-test", "run_id": run_id, "model": "resnet18_cifar", "dataset": f"CIFAR-10 official test set, {int(yt.numel())} images",
               "checkpoint_hash": ck_sha, "codec_version": f"mixed|{E2.SCHEMA_VERSION}"}
        trials_t = {d: [E2.run_mixed_trial(inj, Xt, yt, clean_t, ctx, assignment_id=aid, dist=d, k=DIST_K[d], trial=t, base_seed=cfg.seed) for t in range(args.trials)] for d in dists}
        test_eval = {"assignment_id": aid, "test_samples": int(yt.numel()), "fp32_test_accuracy": fp32_test["accuracy"], "assembled_test_accuracy": clean_t["accuracy"],
                     "test_clean_drop_pp_vs_fp32": fp32_test["accuracy"] - clean_t["accuracy"], "test_fault": {d: E2.summarize_trials(r) for d, r in trials_t.items()},
                     "note": "official test set, evaluated once for the frozen assignment; do not compare with the 1,000-image validation fault screen as the same protocol",
                     "all_restored_exactly": all(r["restored_exactly"] for rs in trials_t.values() for r in rs)}
        dump_json(test_eval, odir / "final_test_evaluation.json")
        protected = any("+" in o for o in opts)
        dump_json({"label": ("JOINT datatype-protection result: the frozen assignment uses the implemented and measured protection scale_tmr on its INT layers. "
                             if protected else "DATATYPE-ONLY result (no protected layer in the frozen assignment). ") + "Frozen from MEASURED end-to-end validation results under the stated constraints.",
                   "assignment_id": aid, "modes_found": asg[aid]["modes"],
                   "selected": [{"layer_id": l, "layer_name": p.layers[l], "num_weights": int(nw[l]), "datatype": P.parse_option(opts[l])[0], "protection": P.parse_option(opts[l])[1],
                                 "Q": float(p.Q[l, choice_of[aid][l]]), "R_exposure_pp": float(p.R[l, choice_of[aid][l]]), "R_scale_word_pp": float(p.Rscale[l, choice_of[aid][l]]),
                                 "S_bytes": float(p.S[l, choice_of[aid][l]])} for l in range(L)],
                   "constraints": {"B_avg_bits_per_weight": args.final_avg_bits, "delta_clean_pp": args.final_delta_clean, "tau_fault_weights_k8_mean_drop_pp": args.final_tau_fault,
                                   "tau_scale_scales_k1_mean_drop_pp": args.final_tau_scale, "A_FP32_validation": fp32["val_accuracy"],
                                   "checked_on": "measured end-to-end validation metrics (not layer-additive proxies)"},
                   "objective": {"formula": "sum x[l,d,p] (Q_norm + lambda R_norm)", "q_column": args.q_column, "r_column": args.r_column, "lambda": args.lam, "normalization": args.norm,
                                 "value": b["objective_value"]},
                   "optimiser_settings_that_produced_it": sw[:3], "solver": "scipy.optimize.milp (HiGHS), exact; candidate assignments from normalization_lambda_sweep.csv; final choice among them by measured feasibility",
                   "measured_validation_screen": {"storage": b["storage"], "clean_validation": b["clean_validation"], "fault_end_to_end": b["fault_end_to_end"]},
                   "measured_validation_final_trials": {"trials_per_distribution": args.final_trials, "fault_end_to_end": final_v},
                   "test_set": {"file": "final_test_evaluation.json", "assembled_test_accuracy": clean_t["accuracy"], "fp32_test_accuracy": fp32_test["accuracy"]}}, odir / "assignment.json")
        print(f"FINAL: {aid} frozen; test accuracy {clean_t['accuracy']:.2f} % (FP32 {fp32_test['accuracy']:.2f} %) -> assignment.json, final_test_evaluation.json")


if __name__ == "__main__":
    main()
