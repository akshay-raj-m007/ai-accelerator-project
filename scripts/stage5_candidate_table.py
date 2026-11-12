"""J-FAMP Stage 5 (7.1): the per-layer x datatype x protection candidate table (DATATYPE-ONLY: protection p = none).

Needs, on the GPU machine: the FP32 checkpoint, the frozen Stage 2 outputs (incl. quantized_weights/), the Stage 4 fault summary, and two per-weight Fishers:
  PRIMARY      exact-expectation Fisher on held-out VALIDATION images 0..2047 (results/<run>/sensitivity_heldout/fisher_exact_val.pt; scripts/fisher_exact.py)
  ALTERNATIVE  sampled-label true Fisher on 1,024 TRAINING images (results/<run>/sensitivity/fisher_true.pt; scripts/fisher_sensitivity.py), kept and labelled
No GPU compute is needed (elementwise arithmetic on 11 M weights x 5 formats), and no images are read. Why the primary changed: phase5/Q_ANALYSIS.md.

  python scripts/stage5_candidate_table.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch

Writes results/<run>/jfamp/optimization/jfamp_candidate_table.csv, candidate_table_meta.json (definitions, input hashes, labels) and
q_validation.json (does sum 0.5*Q predict the measured clean cross-entropy change of the uniform formats?).
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from fr import stage2 as S2
from fr import stage5 as S5
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.model import load_model, quantizable_modules
from fr.sensitivity import load_fisher
from fr.utils import dump_json


def spearman(a, b):
    from fr.layer_scores import spearman as sp
    return sp(a, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fisher", default=None, help="ALTERNATIVE Q: default results/<run>/sensitivity/fisher_true.pt (training-split sampled-label true Fisher)")
    ap.add_argument("--fisher-val", default=None, help="PRIMARY Q: default results/<run>/sensitivity_heldout/fisher_exact_val.pt (exact Fisher, validation images 0..2047)")
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--fault-dir", default=None, help="Stage 4 outputs, default results/<run>/jfamp/fault_screening")
    ap.add_argument("--out-dir", default=None, help="default results/<run>/jfamp/optimization")
    ap.add_argument("--formats", nargs="+", default=S5.FORMATS)
    ap.add_argument("--tmr-dir", default=None, help="folder with tmr_layerwise_summary.csv (scripts/stage5_tmr_layerwise.py), default = --out-dir")
    ap.add_argument("--no-protection", action="store_true", help="datatype-only table (no scale_tmr rows)")
    ap.add_argument("--q-layerwise", default=None, help="q_layerwise_validation.csv for the per-layer measured clean metric (default <out-dir>/q_layerwise_validation.csv)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    fdir = Path(args.fault_dir) if args.fault_dir else base / "fault_screening"
    out = Path(args.out_dir) if args.out_dir else base / "optimization"
    fpath = Path(args.fisher) if args.fisher else cfg.results_dir() / "sensitivity" / "fisher_true.pt"
    fvpath = Path(args.fisher_val) if args.fisher_val else cfg.results_dir() / "sensitivity_heldout" / "fisher_exact_val.pt"
    out.mkdir(parents=True, exist_ok=True)

    man2 = json.load(open(qdir / "quantizer_manifest.json"))
    ck = Path(args.checkpoint)
    ck_sha = sha256_file(ck)
    if man2["checkpoint"]["sha256"] != ck_sha:
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    fres = load_fisher(fpath)
    if fres.get("kind") != "true":
        sys.exit(f"{fpath} is a {fres.get('kind')!r} Fisher; the true Fisher is required")
    if fres.get("source") not in (None, "train"):
        sys.exit(f"{fpath} was computed on the {fres.get('source')!r} split; Q must use training-split calibration images (no validation leakage)")
    F = fres["fisher"]
    fv = load_fisher(fvpath)
    if fv.get("kind") != "exact" or fv.get("source") != "val" or list(fv.get("val_index_range", [])) != [0, int(fv.get("n_samples", -1))]:
        sys.exit(f"{fvpath}: the primary Q needs the exact-expectation Fisher on validation images 0..n-1 (scripts/fisher_exact.py)")
    if fv.get("checkpoint_sha256") != ck_sha:
        sys.exit(f"{fvpath} was computed on a different checkpoint")
    FV = fv["fisher"]
    model, _ = load_model(ck)
    mods = quantizable_modules(model)
    names = [n for n, _ in mods]
    lnames = [f"{n}.weight" for n in names]
    if list(F) != lnames or list(FV) != lnames:
        sys.exit("Fisher layer names / order differ from the model's Conv2d/Linear weights")
    W = {f"{n}.weight": m.weight.detach().cpu().numpy() for n, m in mods}

    fault_rows = list(csv.DictReader(open(fdir / "fault_summary.csv")))
    risk = S5.risk_lookup([r for r in fault_rows])
    codecs = S2.make_codecs(True)
    acc2 = {r["format_key"]: r for r in csv.DictReader(open(qdir / "clean_accuracy.csv"))}
    stor = {r["format_key"]: r for r in csv.DictReader(open(qdir / "storage_summary.csv"))}
    camp = json.load(open(fdir / "fault_campaign_manifest.json"))
    clean1000 = {f: v["clean_accuracy"] for f, v in camp["clean_per_format"].items()}

    rows = []
    for fmt in args.formats:
        stored = S2.load_encoded(qdir / man2["formats"][fmt]["encoded_file"], lnames)
        if S2.content_sha256(stored) != man2["formats"][fmt]["content_sha256"]:
            sys.exit(f"{fmt}: encoded weights differ from the frozen content hash")
        codec = codecs[fmt]
        for lid, (n, ln) in enumerate(zip(names, lnames)):
            what = codec.decode(stored[ln]).reshape(W[ln].shape)
            rt = S2.roundtrip_row(W[ln], what)
            st = S2.storage_row(codec, stored[ln])
            q = S5.fisher_q(W[ln], what, F[ln].numpy())                 # alternative: training-image sampled-label Fisher
            qv = S5.fisher_q(W[ln], what, FV[ln].numpy())               # primary: validation-image exact-expectation Fisher
            row = {"layer_id": lid, "layer_name": n, "num_weights": int(W[ln].size), "datatype": fmt, "datatype_label": S5.LABEL[fmt],
                   "protection": S5.PROTECTION,
                   "Q_fisher_val_exact": qv, "Q_fisher_train_sampled": q, "Q_pred_dCE_val": 0.5 * qv, "Q_pred_dCE_train": 0.5 * q, "relative_l2_error": rt["relative_l2"], "sqnr_db": rt["sqnr_db"],
                   "max_abs_error": rt["max_abs_error"], "decoded_nonfinite": rt["decoded_nonfinite"]}
            row.update(S5.risk_columns(risk, fmt, n))
            row.update({"S_payload_bytes": st["payload_bytes"], "S_scale_bytes": st["scale_bytes"], "S_padding_bits": st["padding_bits"],
                        "S_total_bytes": st["total_bytes"], "bits_per_weight": 8.0 * st["total_bytes"] / W[ln].size,
                        "format_clean_val_accuracy_5000": float(acc2[fmt]["val_accuracy_percent"]),
                        "format_clean_val_delta_ce_5000": float(acc2[fmt]["val_delta_ce"]),
                        "format_clean_accuracy_screen_1000": float(clean1000[fmt]), "S_protection_overhead_bytes": 0})
            rows.append(row)
        print(f"{fmt}: {len(lnames)} layers done")

    # ---- PRIMARY R (declared distribution): exposure-weighted per-flip risk, additive over layers (see stage5.py)
    n_total = int(sum(W[ln].size for ln in lnames))
    for r in rows:
        r["R_per_flip_drop_pp"] = S5.per_flip_drop(r["R_any_k1_mean_drop_pp"], r["R_any_k8_mean_drop_pp"])
        r["R_exposure_k8_pp"] = S5.exposure_risk(r["num_weights"], n_total, r["R_any_k1_mean_drop_pp"], r["R_any_k8_mean_drop_pp"])
    # ---- SECOND metric: scale-word risk (never merged into R). unprotected: Stage 4 layer values, pooled per format over the 21 layers (the layer values are noisy: 20 bimodal trials)
    for fmt in args.formats:
        vals = [r["R_scale_any_k1_mean_drop_pp"] for r in rows if r["datatype"] == fmt]
        pooled = float(np.mean(vals)) if rows and S2.make_codecs(True)[fmt].has_scale else 0.0
        for r in rows:
            if r["datatype"] == fmt:
                r["R_scale_pooled_format_pp"] = pooled
    # ---- measured per-layer clean metric (only layer l quantized, validation images 2,048-4,999), if q_layerwise_validation.csv is available
    qlw = Path(args.q_layerwise) if args.q_layerwise else out / "q_layerwise_validation.csv"
    single = {(r["datatype"], r["layer_name"]): r for r in csv.DictReader(open(qlw, encoding="utf8"))} if qlw.exists() else {}
    for r in rows:
        s = single.get((r["datatype"], r["layer_name"]))
        r["clean_single_layer_dCE_val"] = float(s["measured_dCE"]) if s else ""
        r["clean_single_layer_acc_drop_pp_val"] = float(s["measured_acc_drop_pp"]) if s else ""
        r["option"] = r["datatype"]
    # ---- protection option scale_tmr for the INT formats: S + 8 bytes, weights risk unchanged, scale risk MEASURED on the protected store
    tmr_rows = []
    if not args.no_protection:
        tp = Path(args.tmr_dir) if args.tmr_dir else out
        tsum = tp / "tmr_layerwise_summary.csv"
        if not tsum.exists():
            sys.exit(f"{tsum} not found: run scripts/stage5_tmr_layerwise.py first (the protection option must be measured, not assumed), or pass --no-protection")
        tmr = {(r["format"], r["layer_name"]): r for r in csv.DictReader(open(tsum, encoding="utf8"))}
        pooled_tmr = {f: float(np.mean([float(r["protected_drop_pp_mean"]) for (ff, _), r in tmr.items() if ff == f])) for f in ("int8_pt", "int4_pt")}
        for r in rows:
            if r["datatype"] not in ("int8_pt", "int4_pt"):
                continue
            t = tmr[(r["datatype"], r["layer_name"])]            # layer_name = module name (e.g. layer1.0.conv1)
            q = dict(r)
            q.update({"protection": "scale_tmr", "option": f"{r['datatype']}+scale_tmr", "datatype_label": r["datatype_label"] + " + scale TMR",
                      "S_protection_overhead_bytes": 8, "S_total_bytes": r["S_total_bytes"] + 8,
                      "bits_per_weight": 8.0 * (r["S_total_bytes"] + 8) / r["num_weights"],
                      "R_scale_any_k1_mean_drop_pp": float(t["protected_drop_pp_mean"]), "R_scale_pooled_format_pp": pooled_tmr[r["datatype"]],
                      "R_scale_unprotected_control_pp": float(t["unprotected_control_drop_pp_mean"]), "R_scale_protected_worst_pp": float(t["protected_drop_pp_worst"]),
                      "R_scale_n_trials": int(t["n_trials"])})
            tmr_rows.append(q)
    for r in rows:
        r.setdefault("R_scale_unprotected_control_pp", "")
        r.setdefault("R_scale_protected_worst_pp", "")
        r.setdefault("R_scale_n_trials", int(r["R_n_trials"]) if r["has_scale_word"] else "")
    for q in tmr_rows:
        q.setdefault("R_scale_unprotected_control_pp", "")
    rows = rows + tmr_rows

    fields = list(rows[0])
    with open(out / "jfamp_candidate_table.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # does the Fisher-weighted Q predict the clean change of the uniform formats? (5 points, reported not tuned; measured on ALL 5,000 validation images, which include the
    # 2,048 Fisher images - the strict disjoint check is scripts/stage5_q_validation.py)
    val = []
    for fmt in args.formats:
        val.append({"datatype": fmt, "predicted_dCE_val_fisher": sum(r["Q_pred_dCE_val"] for r in rows if r["datatype"] == fmt and r["protection"] == "none"),
                    "predicted_dCE_train_fisher": sum(r["Q_pred_dCE_train"] for r in rows if r["datatype"] == fmt and r["protection"] == "none"),
                    "measured_val_delta_ce_5000": float(acc2[fmt]["val_delta_ce"]), "measured_val_drop_pp_5000": float(acc2[fmt]["val_drop_pp"])})
    meas = [v["measured_val_delta_ce_5000"] for v in val]
    sp_v = spearman([v["predicted_dCE_val_fisher"] for v in val], meas)
    sp_t = spearman([v["predicted_dCE_train_fisher"] for v in val], meas)
    dump_json({"note": "uniform assignments; 1/2 sum Q (second-order estimate) vs the measured clean CE change on 5,000 validation images (overlaps the Fisher images; see q_layerwise_validation.json for the disjoint check)",
               "rows": val, "spearman_val_fisher_vs_measured": sp_v, "spearman_train_fisher_vs_measured": sp_t,
               "measured_values_for_the_four_non_INT4_formats_are_within_evaluation_noise": True}, out / "q_validation.json")

    const = float(stor[args.formats[0]]["non_quantized_param_bytes_fp32"]) + float(stor[args.formats[0]]["buffer_bytes"])
    meta = {"label": ("DATATYPE-ONLY reliability-aware analysis (protection p = none)" if args.no_protection else
                      "JOINT datatype-protection analysis: protection options none and scale_tmr (INT scale word stored 3x, majority vote), both implemented and measured; the datatype-only analysis is the p = none subset"),
            "options": sorted({r["option"] for r in rows}), "R_primary_column": "R_exposure_k8_pp", "R_round1_column": "R_any_k8_mean_drop_pp",
            "R_scale_columns": "R_scale_pooled_format_pp (used by the optimiser), R_scale_any_k1_mean_drop_pp (per layer)",
            "Q_definition": S5.Q_DEFINITION, "Q_primary_column": "Q_fisher_val_exact", "Q_alternative_column": "Q_fisher_train_sampled", "R_definition": S5.R_DEFINITION, "formats": args.formats, "layers": names,
            "fixed_non_quantized_bytes": const, "num_quantized_weights": int(sum(W[ln].size for ln in lnames)),
            "inputs": {"checkpoint_sha256": ck_sha, "fisher_file": str(fpath), "fisher_sha256": sha256_file(fpath), "fisher_kind": fres.get("kind"),
                       "fisher_source": fres.get("source"), "fisher_n_samples": fres.get("n_samples"),
                       "fisher_val_file": str(fvpath), "fisher_val_sha256": sha256_file(fvpath), "fisher_val_kind": fv.get("kind"), "fisher_val_source": fv.get("source"),
                       "fisher_val_n_samples": fv.get("n_samples"), "fisher_val_index_range": fv.get("val_index_range"), "fisher_val_min_prob": fv.get("min_prob"),
                       "stage2_frozen_config_sha256": man2["frozen_config_sha256"],
                       "stage4_fault_summary_sha256": sha256_file(fdir / "fault_summary.csv"),
                       "stage4_fault_trials_sha256": sha256_file(fdir / "fault_trials.jsonl"), "stage4_run_id": camp["run_id"]},
            "notes": ["Layer-additive scores approximate the full model; the selected assignment must be assembled and measured end to end.",
                      "R_scale_* is a different fault process (one flipped bit of an INT scale word) and is a SECOND metric, never merged into the primary R.",
                      "Clean accuracy: format-level (Stage 2) and, per layer, the single-layer measurement clean_single_layer_* (validation images 2,048-4,999, q_layerwise_validation.csv); the complete-model clean accuracy is measured end to end."]}
    dump_json(meta, out / "candidate_table_meta.json")
    print(f"{len(rows)} candidate rows ({len(tmr_rows)} with scale_tmr) -> {out / 'jfamp_candidate_table.csv'}")
    print(f"Spearman (sum 0.5 Q vs measured val dCE, 5 uniform formats): val-Fisher {sp_v:+.3f}, train-Fisher {sp_t:+.3f}")
    for v in val:
        print(f"  {v['datatype']:11s} predicted dCE: val-Fisher {v['predicted_dCE_val_fisher']:.2e}  train-Fisher {v['predicted_dCE_train_fisher']:.2e}   measured {v['measured_val_delta_ce_5000']:+.5f}")


if __name__ == "__main__":
    main()
