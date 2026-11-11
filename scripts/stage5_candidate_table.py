"""J-FAMP Stage 5 (7.1): the per-layer x datatype x protection candidate table (DATATYPE-ONLY: protection p = none).

Needs, on the GPU machine: the FP32 checkpoint, the frozen Stage 2 outputs (incl. quantized_weights/), the Stage 4 fault summary, and the
per-weight true Fisher computed on the TRAINING split (results/<run>/sensitivity/fisher_true.pt; scripts/fisher_sensitivity.py). No GPU compute is
needed (elementwise arithmetic on 11 M weights x 5 formats), and no images are read.

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
    ap.add_argument("--fisher", default=None, help="default results/<run>/sensitivity/fisher_true.pt (training-split true Fisher)")
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--fault-dir", default=None, help="Stage 4 outputs, default results/<run>/jfamp/fault_screening")
    ap.add_argument("--out-dir", default=None, help="default results/<run>/jfamp/optimization")
    ap.add_argument("--formats", nargs="+", default=S5.FORMATS)
    args = ap.parse_args()
    cfg = config_from_args(args)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    fdir = Path(args.fault_dir) if args.fault_dir else base / "fault_screening"
    out = Path(args.out_dir) if args.out_dir else base / "optimization"
    fpath = Path(args.fisher) if args.fisher else cfg.results_dir() / "sensitivity" / "fisher_true.pt"
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
    model, _ = load_model(ck)
    mods = quantizable_modules(model)
    names = [n for n, _ in mods]
    lnames = [f"{n}.weight" for n in names]
    if list(F) != lnames:
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
            q = S5.fisher_q(W[ln], what, F[ln].numpy())
            row = {"layer_id": lid, "layer_name": n, "num_weights": int(W[ln].size), "datatype": fmt, "datatype_label": S5.LABEL[fmt],
                   "protection": S5.PROTECTION,
                   "Q_fisher_weighted": q, "Q_pred_dCE": 0.5 * q, "relative_l2_error": rt["relative_l2"], "sqnr_db": rt["sqnr_db"],
                   "max_abs_error": rt["max_abs_error"], "decoded_nonfinite": rt["decoded_nonfinite"]}
            row.update(S5.risk_columns(risk, fmt, n))
            row.update({"S_payload_bytes": st["payload_bytes"], "S_scale_bytes": st["scale_bytes"], "S_padding_bits": st["padding_bits"],
                        "S_total_bytes": st["total_bytes"], "bits_per_weight": 8.0 * st["total_bytes"] / W[ln].size,
                        "format_clean_val_accuracy_5000": float(acc2[fmt]["val_accuracy_percent"]),
                        "format_clean_val_delta_ce_5000": float(acc2[fmt]["val_delta_ce"]),
                        "format_clean_accuracy_screen_1000": float(clean1000[fmt]), "R_protection_overhead_bytes": 0})
            rows.append(row)
        print(f"{fmt}: {len(lnames)} layers done")

    fields = list(rows[0])
    with open(out / "jfamp_candidate_table.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # does the Fisher-weighted Q predict the clean change of the uniform formats? (5 points, reported not tuned)
    val = []
    for fmt in args.formats:
        pred = sum(r["Q_pred_dCE"] for r in rows if r["datatype"] == fmt)
        val.append({"datatype": fmt, "predicted_dCE_sum_half_Q": pred, "measured_val_delta_ce_5000": float(acc2[fmt]["val_delta_ce"]),
                    "measured_val_drop_pp_5000": float(acc2[fmt]["val_drop_pp"])})
    sp = spearman([v["predicted_dCE_sum_half_Q"] for v in val], [v["measured_val_delta_ce_5000"] for v in val])
    dump_json({"note": "uniform assignments; Q is a second-order estimate from training-split Fisher, measured values are on 5,000 validation images",
               "rows": val, "spearman_predicted_vs_measured_dCE": sp}, out / "q_validation.json")

    const = float(stor[args.formats[0]]["non_quantized_param_bytes_fp32"]) + float(stor[args.formats[0]]["buffer_bytes"])
    meta = {"label": "DATATYPE-ONLY reliability-aware analysis (protection p = none; no protection implemented)",
            "Q_definition": S5.Q_DEFINITION, "R_definition": S5.R_DEFINITION, "formats": args.formats, "layers": names,
            "fixed_non_quantized_bytes": const, "num_quantized_weights": int(sum(W[ln].size for ln in lnames)),
            "inputs": {"checkpoint_sha256": ck_sha, "fisher_file": str(fpath), "fisher_sha256": sha256_file(fpath), "fisher_kind": fres.get("kind"),
                       "fisher_source": fres.get("source"), "fisher_n_samples": fres.get("n_samples"),
                       "stage2_frozen_config_sha256": man2["frozen_config_sha256"],
                       "stage4_fault_summary_sha256": sha256_file(fdir / "fault_summary.csv"),
                       "stage4_fault_trials_sha256": sha256_file(fdir / "fault_trials.jsonl"), "stage4_run_id": camp["run_id"]},
            "notes": ["Layer-additive scores approximate the full model; the selected assignment must be assembled and measured end to end.",
                      "R_scale_any_k1_mean_drop_pp is a different fault process (INT scale words, k = 1) and is not part of R unless a sweep variant says so.",
                      "Clean accuracy is a format-level measurement (Stage 2, all layers in the same format); per-layer clean accuracy was not measured."]}
    dump_json(meta, out / "candidate_table_meta.json")
    print(f"{len(rows)} candidate rows -> {out / 'jfamp_candidate_table.csv'}")
    print(f"Spearman (sum 0.5 Q vs measured val dCE, 5 uniform formats): {sp:+.3f}")
    for v in val:
        print(f"  {v['datatype']:11s} predicted dCE {v['predicted_dCE_sum_half_Q']:.5f}   measured {v['measured_val_delta_ce_5000']:+.5f}")


if __name__ == "__main__":
    main()
