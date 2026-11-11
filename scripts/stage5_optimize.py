"""J-FAMP Stage 5 (7.2): normalisation, lambda sweep and constrained assignment (ILP) on the candidate table. CPU only, seconds.

DATATYPE-ONLY analysis (protection p = none). Reads jfamp_candidate_table.csv (stage5_candidate_table.py) and writes
  normalization_lambda_sweep.csv   every solved setting: normalisation, lambda, constraints, status, objective, storage, predicted clean/fault proxies, assignment
  distinct_assignments.csv         the distinct optimal assignments found in the sweep with how often each occurs
  assignment.json                  the assignment of ONE operating point (given by --select-*; a PLACEHOLDER until the constraints are agreed)
  optimization_report.md

Objective  min  sum_{l,d} x[l,d] (Q_norm[l,d] + lambda R_norm[l,d])   s.t.  one datatype per layer; sum S <= B; mean_l R <= tau_fault;
sum_l 0.5 Q <= eps_CE (predicted clean CE increase); x binary.   Solver: scipy.optimize.milp (HiGHS), exact.
Normalisation (global over all candidates): "minmax" -> (v - min)/(max - min), "rank" -> average rank / (n-1).
Risk variants: "weights" = R as defined in the table (Stage 4, any bit, k = 8); "plus_scale" = R + the Stage 4 INT scale-word risk (k = 1), a union of two
different fault processes used only as a sensitivity check.

  python scripts/stage5_optimize.py --run-name scratch      (or --table PATH --out-dir DIR)
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import scipy
from fr import stage5 as S5
from fr.config import add_config_args, config_from_args
from fr.utils import dump_json

SHORT = {"e4m11_sn": "E4M11", "fp16": "FP16", "fp8_e4m3fn": "FP8", "int8_pt": "INT8", "int4_pt": "INT4"}


def load_problem(path):
    rows = list(csv.DictReader(open(path, encoding="utf8")))
    layers, formats = [], []
    for r in rows:
        if r["layer_name"] not in layers:
            layers.append(r["layer_name"])
        if r["datatype"] not in formats:
            formats.append(r["datatype"])
    L, D = len(layers), len(formats)
    arr = {k: np.zeros((L, D)) for k in ("Q", "R", "S", "Rs", "n")}
    seen = set()
    for r in rows:
        i, j = layers.index(r["layer_name"]), formats.index(r["datatype"])
        if (i, j) in seen:
            raise SystemExit("duplicate candidate row")
        seen.add((i, j))
        if r["protection"] != "none":
            raise SystemExit("this optimiser is datatype-only: only protection = none is allowed")
        arr["Q"][i, j] = float(r["Q_fisher_weighted"])
        arr["R"][i, j] = float(r["R_any_k8_mean_drop_pp"])
        arr["S"][i, j] = float(r["S_total_bytes"]) + float(r["R_protection_overhead_bytes"])
        arr["Rs"][i, j] = float(r["R_scale_any_k1_mean_drop_pp"])
        arr["n"][i, j] = float(r["num_weights"])
    if len(seen) != L * D:
        raise SystemExit("candidate table is not a full layer x datatype grid")
    p = S5.Problem(layers, formats, arr["Q"], arr["R"], arr["S"], Rscale=arr["Rs"])
    return p, arr["n"][:, 0], rows


def fnum(x, n=6):
    return "" if x is None else f"{x:.{n}g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--table", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1, 2, 4, 10])
    ap.add_argument("--avg-bits", type=float, nargs="+", default=[4.5, 5, 6, 7, 8, 10, 12, 16], help="storage budget B as average stored bits per weight")
    ap.add_argument("--tau", type=float, nargs="*", default=[5, 2, 1, 0.5], help="max mean layer risk in pp (a run without this constraint is always added)")
    ap.add_argument("--eps-ce", type=float, nargs="*", default=[], help="max predicted clean CE increase (a run without it is always added). Off by default: with the Fisher Q scale (see q_validation.json) every solution predicts <= 1.4e-6, so any realistic limit is vacuous")
    ap.add_argument("--norms", nargs="+", default=["minmax", "rank"])
    ap.add_argument("--risks", nargs="+", default=["weights", "plus_scale"])
    ap.add_argument("--select-norm", default="minmax")
    ap.add_argument("--select-risk", default="weights")
    ap.add_argument("--select-lambda", type=float, default=1.0)
    ap.add_argument("--select-bits", type=float, default=8.0)
    ap.add_argument("--select-tau", type=float, default=None)
    ap.add_argument("--select-eps", type=float, default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)
    base = cfg.results_dir() / "jfamp" / "optimization"
    table = Path(args.table) if args.table else base / "jfamp_candidate_table.csv"
    out = Path(args.out_dir) if args.out_dir else table.parent
    out.mkdir(parents=True, exist_ok=True)
    meta_path = table.parent / "candidate_table_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    const_bytes = float(meta.get("fixed_non_quantized_bytes", 0.0))

    p, nw, _ = load_problem(table)
    N = float(nw.sum())
    L, D = len(p.layers), len(p.formats)
    print(f"{L} layers x {D} datatypes = {L * D} candidates, {int(N):,} quantized weights")

    taus = [None] + list(args.tau)
    epss = [None] + list(args.eps_ce)
    rows, n_solved = [], 0
    for norm in args.norms:
        for risk in args.risks:
            for lam in args.lambdas:
                for bits in args.avg_bits:
                    for tau in taus:
                        for eps in epss:
                            B = bits * N / 8.0
                            sol = S5.solve(p, lam, norm, risk, max_bytes=B, max_mean_R=tau, max_dce=eps)
                            row = {"normalization": norm, "risk_variant": risk, "lambda": lam, "B_avg_bits_per_weight": bits, "B_bytes": B,
                                   "tau_fault_mean_layer_risk_pp": "" if tau is None else tau, "eps_clean_pred_dCE": "" if eps is None else eps,
                                   "status": sol["status"], "objective": "", "storage_bytes": "", "avg_bits_per_weight": "",
                                   "model_storage_bytes_incl_fixed": "", "sum_Q": "", "pred_clean_dCE": "", "mean_layer_risk_pp": "",
                                   "mean_layer_risk_plus_scale_pp": "", "n_e4m11": "", "n_fp16": "", "n_fp8": "", "n_int8": "", "n_int4": "", "assignment": ""}
                            if sol["choice"] is not None:
                                t = p.totals(sol["choice"])
                                Rv = p.R if risk == "weights" else p.R + p.Rscale
                                assert t["storage_bytes"] <= B + 1e-6
                                if tau is not None:
                                    assert Rv[np.arange(L), sol["choice"]].mean() <= tau + 1e-9
                                if eps is not None:
                                    assert t["predicted_dCE"] <= eps + 1e-12
                                cnt = Counter(p.formats[c] for c in sol["choice"])
                                row.update({"objective": fnum(sol["objective"], 10), "storage_bytes": int(t["storage_bytes"]),
                                            "avg_bits_per_weight": fnum(8 * t["storage_bytes"] / N), "model_storage_bytes_incl_fixed": int(t["storage_bytes"] + const_bytes),
                                            "sum_Q": fnum(t["sum_Q"], 8), "pred_clean_dCE": fnum(t["predicted_dCE"], 8), "mean_layer_risk_pp": fnum(t["mean_R_pp"]),
                                            "mean_layer_risk_plus_scale_pp": fnum(t["mean_R_plus_scale_pp"]),
                                            **{f"n_{k}": cnt.get(f, 0) for k, f in (("e4m11", "e4m11_sn"), ("fp16", "fp16"), ("fp8", "fp8_e4m3fn"), ("int8", "int8_pt"), ("int4", "int4_pt"))},
                                            "assignment": ";".join(SHORT.get(p.formats[c], p.formats[c]) for c in sol["choice"])})
                            rows.append(row)
                            n_solved += 1
    with open(out / "normalization_lambda_sweep.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "optimal"]
    print(f"{n_solved} problems solved: {len(ok)} optimal, {n_solved - len(ok)} infeasible")

    distinct = Counter(r["assignment"] for r in ok)
    with open(out / "distinct_assignments.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["assignment_id", "n_settings", "storage_bytes", "avg_bits_per_weight", "sum_Q", "pred_clean_dCE", "mean_layer_risk_pp", "mean_layer_risk_plus_scale_pp", "assignment"])
        for i, (a, c) in enumerate(distinct.most_common()):
            ch = [next(k for k, v in SHORT.items() if v == s) for s in a.split(";")]
            t = p.totals([p.formats.index(c_) for c_ in ch])
            w.writerow([f"A{i:03d}", c, int(t["storage_bytes"]), fnum(8 * t["storage_bytes"] / N), fnum(t["sum_Q"], 8), fnum(t["predicted_dCE"], 8),
                        fnum(t["mean_R_pp"]), fnum(t["mean_R_plus_scale_pp"]), a])
    print(f"{len(distinct)} distinct optimal assignments")

    # ---- the selected operating point (placeholder; not tuned)
    B = args.select_bits * N / 8.0
    sol = S5.solve(p, args.select_lambda, args.select_norm, args.select_risk, max_bytes=B, max_mean_R=args.select_tau, max_dce=args.select_eps)
    if sol["choice"] is None:
        sys.exit(f"the selected operating point is {sol['status']}")
    t = p.totals(sol["choice"])
    layers = []
    for l, c in enumerate(sol["choice"]):
        layers.append({"layer_id": l, "layer_name": p.layers[l], "num_weights": int(nw[l]), "datatype": p.formats[c], "protection": "none",
                       "Q": p.Q[l, c], "R_any_k8_mean_drop_pp": p.R[l, c], "R_scale_any_k1_mean_drop_pp": p.Rscale[l, c], "S_bytes": p.S[l, c]})
    dump_json({"label": "DATATYPE-ONLY analysis (protection p = none, nothing implemented). PROVISIONAL operating point: constraints are not agreed yet and "
                        "were not tuned. Layer-additive estimates only; must be measured end to end.",
               "selected": layers,
               "constraints": {"B_avg_bits_per_weight": args.select_bits, "B_bytes": B, "tau_fault_mean_layer_risk_pp": args.select_tau,
                               "eps_clean_pred_dCE": args.select_eps, "delta_clean": "checked end to end after assembly (not an optimiser constraint)"},
               "objective": {"formula": "sum x[l,d] (Q_norm + lambda R_norm)", "lambda": args.select_lambda, "normalization": args.select_norm,
                             "risk_variant": args.select_risk, "value": sol["objective"]},
               "totals_predicted": t, "model_storage_bytes_incl_fixed": t["storage_bytes"] + const_bytes,
               "solver": {"name": "scipy.optimize.milp (HiGHS), exact", "scipy": scipy.__version__, "variables": L * D},
               "R_definition": S5.R_DEFINITION, "Q_definition": S5.Q_DEFINITION,
               "candidate_table_sha256": meta.get("inputs", {})}, out / "assignment.json")

    # ---- report
    lines = ["# Stage 5 optimisation (datatype-only, p = none)", "",
             f"{n_solved} problems solved ({len(ok)} optimal), {len(distinct)} distinct optimal assignments; full sweep in `normalization_lambda_sweep.csv`.", "",
             "## Selected operating point (provisional placeholder)", "",
             f"B = {args.select_bits} avg bits/weight, lambda = {args.select_lambda}, normalisation {args.select_norm}, risk variant {args.select_risk}, "
             f"tau = {args.select_tau}, eps = {args.select_eps}.", "",
             "| layer | datatype | Q | R (pp) | bytes |", "|---|---|---|---|---|"]
    lines += [f"| {x['layer_name']} | {S5.LABEL[x['datatype']]} | {x['Q']:.3g} | {x['R_any_k8_mean_drop_pp']:.2f} | {int(x['S_bytes']):,} |" for x in layers]
    lines += ["", f"Totals (layer-additive prediction): storage {int(t['storage_bytes']):,} bytes ({8 * t['storage_bytes'] / N:.2f} bits/weight), "
              f"predicted clean CE increase {t['predicted_dCE']:.5f}, mean layer risk {t['mean_R_pp']:.2f} pp. Not measured end to end yet."]
    open(out / "optimization_report.md", "w", encoding="utf8").write("\n".join(lines) + "\n")
    print(f"selected: {t['storage_bytes']:,.0f} bytes, predicted dCE {t['predicted_dCE']:.5f}, mean R {t['mean_R_pp']:.2f} pp -> {out / 'assignment.json'}")


if __name__ == "__main__":
    main()
