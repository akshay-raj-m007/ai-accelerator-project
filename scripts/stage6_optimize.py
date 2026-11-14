#!/usr/bin/env python
"""MobileNetV2 protocol Section 6, part 1 (CPU, minutes): candidate table, normalisation / lambda sweep and the constrained binary assignment.

Reads ONLY files that are already in the repository (Stage 3 layer-wise sensitivity, Stage 4 layer-wise fault summary, Stage 5 protection results, the
Stage 6 scale-word campaign) and the frozen configuration hashes in their manifests; no model, no data, no GPU.

  python scripts/stage6_optimize.py --reports reports/mobilenetv2_cifar10

Objective and constraints (same binary assignment as the ResNet-18 JFAMP run; constants = the ResNet-18 optimiser's agreed operating point, taken from its code
defaults scripts/stage5_optimize.py because the saved summary does not list them):
    minimise  sum x[l,d,p] (Qn[l,d] + lambda Rn[l,d,p])      Qn, Rn = global min-max normalisation over ALL candidates, lambda = 1.0
    s.t.      one (d,p) per layer;  sum x S <= B  (B = 8.0 average bits per QUANTIZED Conv/Linear weight = 2,202,560 bytes, quantized weights only, all stored bytes
              counted: payload + scales + alignment + redundancy);  sum x R <= tau_fault = 0.5 pp (predicted mean accuracy drop under 8 random weight-bit flips);
              options whose scale-word risk R_scale > tau_scale = 1.0 pp are inadmissible;  x[l,d,p] = 0 for unsupported combinations (they do not exist).
    A_clean(x) >= A_FP32 - delta_clean (delta_clean = 0.5 pp) and the MEASURED fault risk are checked on the ASSEMBLED model by stage6_end_to_end.py, not inside the ILP.
A second operating point, B = 60 % of the FP32 weight bytes (4.8 bits / weight), is part of the sweep for comparison with the CIFAR-100 run (label: friend's setting).

Writes into --out-dir (default <reports>/stage_06_optimization): jfamp_candidate_table.csv (1,590 rows), candidate_table_meta.json, normalization_lambda_sweep.csv,
distinct_assignments.csv, assignment_predicted.json, declared_constants.json, optimization_report.md.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import scipy
from fr import stage3 as S3
from fr import stage5 as S5
from fr import stage6 as S6
from fr.utils import dump_json, git_info

DECLARED = {"lambda": 1.0, "normalization": "minmax", "B_avg_bits_per_weight": 8.0, "tau_fault_pp": 0.5, "tau_scale_pp": 1.0, "delta_clean_pp": 0.5,
            "source": "ResNet-18 JFAMP optimiser defaults (scripts/stage5_optimize.py: --select-lambda 1.0, --select-norm minmax, --select-bits 8.0, --select-tau 0.5, "
                      "--tau-scale 1.0; delta_clean 0.5 in its assignment.json). Protocol continuation, not an exact numeric reproduction.",
            "B_scope": "quantized Conv2d/Linear weights only (2,202,560 weights); biases, BatchNorm and buffers are FP32 and not counted",
            "friend_setting_for_comparison": {"B_fraction_of_fp32_weight_bytes": 0.6, "B_avg_bits_per_weight": 4.8, "lambda": 1.0}}
LAMBDAS = [0, 0.1, 0.25, 0.5, 1, 2, 4, 10]
AVG_BITS = [4.5, 4.8, 5, 6, 7, 8, 10, 12, 16]
TAUS = [2, 1, 0.5, 0.25]


def read_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", default="reports/mobilenetv2_cifar10")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    ap.add_argument("--avg-bits", type=float, nargs="+", default=AVG_BITS)
    ap.add_argument("--tau", type=float, nargs="*", default=TAUS)
    ap.add_argument("--norms", nargs="+", default=["minmax", "rank"])
    ap.add_argument("--no-protection", action="store_true", help="datatype-only run only")
    args = ap.parse_args(argv)
    rep = Path(args.reports)
    out = Path(args.out_dir) if args.out_dir else rep / "stage_06_optimization"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    paths = {"stage3": rep / "stage_03_layerwise_sensitivity/layerwise_sensitivity_results.csv", "stage4": rep / "stage_04_encoded_fault_injection/layerwise_summary.csv",
             "storage": rep / "stage_05_protection/protection_storage.csv", "prot_layer": rep / "stage_05_protection/protection_layer_summary.csv",
             "prot_summary": rep / "stage_05_protection/protection_summary.csv", "scale": rep / "stage_06_optimization/scale_faults/scale_summary.csv"}
    missing = [k for k, v in paths.items() if not v.exists()]
    if missing:
        sys.exit(f"missing inputs: {[str(paths[k]) for k in missing]} (the scale campaign is stage6_scale_faults.py)")
    # ---- provenance: every input must come from the same checkpoint and frozen quantizer configuration
    frozen = {"stage3": json.load(open(rep / "stage_03_layerwise_sensitivity/stage3_progress.json")), "stage4": json.load(open(rep / "stage_04_encoded_fault_injection/layerwise_manifest.json")),
              "stage5": json.load(open(rep / "stage_05_protection/protection_report.json")), "scale": json.load(open(rep / "stage_06_optimization/scale_faults/scale_manifest.json"))}
    cks = {k: v["checkpoint_sha256"] for k, v in frozen.items()}
    fz = {k: v["quantizer_frozen_config_sha256"] if k == "stage3" else v["stage2_frozen_config_sha256"] for k, v in frozen.items()}
    if len(set(cks.values())) != 1 or len(set(fz.values())) != 1:
        sys.exit(f"inputs come from different checkpoints / frozen configurations: {cks} {fz}")
    ck_sha, fz_sha = next(iter(cks.values())), next(iter(fz.values()))
    rows, meta = S6.build_candidates(read_csv(paths["stage3"]), read_csv(paths["stage4"]), read_csv(paths["storage"]), read_csv(paths["prot_layer"]),
                                     read_csv(paths["prot_summary"]), read_csv(paths["scale"]))
    write_csv(out / "jfamp_candidate_table.csv", rows)
    modes = ["datatype_only"] if args.no_protection else ["datatype_only", "joint"]
    problems = {m: S6.rows_to_problem(rows, m) for m in modes}
    N = float(next(iter(problems.values()))[1].sum())
    for m, (p, _) in problems.items():
        print(f"{m}: {len(p.layers)} layers x {len(p.formats)} options = {len(p.layers) * len(p.formats)} candidates, {int(N):,} quantized weights")
    meta.update({"checkpoint_sha256": ck_sha, "stage2_frozen_config_sha256": fz_sha, "inputs": {k: {"path": str(v), "sha256": hashlib.sha256(open(v, "rb").read()).hexdigest()} for k, v in paths.items()},
                 "source_commit": git_info(), "declared_constants": DECLARED})
    dump_json(meta, out / "candidate_table_meta.json")
    dump_json(DECLARED, out / "declared_constants.json")
    sweep = S6.run_sweep(problems, args.lambdas, args.avg_bits, args.tau, args.norms, DECLARED["tau_scale_pp"])
    write_csv(out / "normalization_lambda_sweep.csv", sweep)
    ok = [r for r in sweep if r["status"] == "optimal"]
    print(f"{len(sweep)} problems solved: {len(ok)} optimal, {len(sweep) - len(ok)} infeasible ({time.time() - t0:.0f} s)")
    pj = problems.get("joint", problems["datatype_only"])[0]
    dist = S6.distinct_assignments(sweep)
    for d in dist:
        ch = [pj.formats.index(S6.long_token(s)) for s in d["assignment"].split(";")]
        t = pj.totals(ch)
        d.update({"storage_bytes": int(t["storage_bytes"]), "avg_bits_per_weight": 8 * t["storage_bytes"] / N, "sum_Q": t["sum_Q"], "pred_weight_fault_drop_pp": t["sum_R_pp"],
                  "pred_max_scale_word_risk_pp": t["max_layer_scale_risk_pp"], "n_protected_layers": sum(1 for c in ch if "+" in pj.formats[c]),
                  "objective_lambda1_minmax": S6.objective_of(pj, ch, 1.0, "minmax")})
    write_csv(out / "distinct_assignments.csv", dist, ["assignment_id", "found_in_modes", "n_settings", "storage_bytes", "avg_bits_per_weight", "sum_Q", "pred_weight_fault_drop_pp",
                                                        "pred_max_scale_word_risk_pp", "n_protected_layers", "objective_lambda1_minmax", "assignment"])
    print(f"{len(dist)} distinct optimal assignments")
    # ---- the PREDICTED selection at the declared operating point (layer-additive; the frozen assignment comes from stage6_end_to_end.py)
    sel_mode = "datatype_only" if args.no_protection else "joint"
    p, nw = problems[sel_mode]
    B = DECLARED["B_avg_bits_per_weight"] * N / 8.0
    sol = S5.solve(p, DECLARED["lambda"], DECLARED["normalization"], "weights", max_bytes=B, max_sum_R=DECLARED["tau_fault_pp"],
                   tau_scale=DECLARED["tau_scale_pp"] if sel_mode == "joint" else None)
    layers_out, t = [], None
    if sol["choice"] is not None:
        t = p.totals(sol["choice"])
        for l, c in enumerate(sol["choice"]):
            f, pr = S6.parse_option(p.formats[c])
            layers_out.append({"layer_id": l, "layer_name": p.layers[l], "num_weights": int(nw[l]), "datatype": f, "protection": pr, "Q": p.Q[l, c], "R_exposure_pp": p.R[l, c],
                               "R_scale_word_pp": p.Rscale[l, c], "S_bytes": p.S[l, c]})
    dump_json({"label": "PREDICTED selection from layer-additive estimates at the declared operating point; NOT measured end to end. The frozen assignment is written by "
                        "stage6_end_to_end.py --finalize after assembly and measurement.", "mode": sel_mode, "status": sol["status"], "selected": layers_out,
               "constraints": {k: DECLARED[k] for k in ("B_avg_bits_per_weight", "tau_fault_pp", "tau_scale_pp", "delta_clean_pp")} | {"B_bytes": B},
               "objective": {"lambda": DECLARED["lambda"], "normalization": DECLARED["normalization"], "value": sol.get("objective")}, "totals_predicted": t,
               "solver": {"name": "scipy.optimize.milp (HiGHS), exact", "scipy": scipy.__version__}, "checkpoint_sha256": ck_sha}, out / "assignment_predicted.json")
    lines = ["# Stage 6 optimisation (part 1, predicted)", "", f"{len(sweep)} problems solved ({len(ok)} optimal), {len(dist)} distinct optimal assignments; full sweep in `normalization_lambda_sweep.csv`.", "",
             f"Declared operating point: B = {DECLARED['B_avg_bits_per_weight']} bits/weight ({int(B):,} bytes, quantized weights only), lambda = {DECLARED['lambda']}, min-max, tau_fault = "
             f"{DECLARED['tau_fault_pp']} pp (predicted), tau_scale = {DECLARED['tau_scale_pp']} pp, delta_clean = {DECLARED['delta_clean_pp']} pp (measured end to end).", ""]
    if t is None:
        lines.append(f"The declared operating point is **{sol['status']}** on the layer-additive estimates.")
    else:
        lines += ["| layer | datatype | protection | Q (CE) | R exposure (pp) | scale-word risk (pp) | bytes |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {x['layer_name']} | {S5.LABEL.get(x['datatype'], x['datatype'])} | {x['protection']} | {x['Q']:.3g} | {x['R_exposure_pp']:.3f} | {x['R_scale_word_pp']:.2f} | {int(x['S_bytes']):,} |" for x in layers_out]
        lines += ["", f"Totals (layer-additive prediction): {int(t['storage_bytes']):,} bytes ({8 * t['storage_bytes'] / N:.3f} bits/weight), predicted CE increase {t['predicted_dCE']:.4f}, predicted "
                      f"weight-fault drop {t['sum_R_pp']:.3f} pp, largest scale-word risk {t['max_layer_scale_risk_pp']:.2f} pp."]
    open(out / "optimization_report.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("predicted selection:", sol["status"], "" if t is None else f"{int(t['storage_bytes']):,} bytes, predicted fault drop {t['sum_R_pp']:.3f} pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
