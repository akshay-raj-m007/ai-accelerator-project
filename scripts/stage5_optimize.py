"""J-FAMP Stage 5 (7.2): normalisation, lambda sweep and constrained assignment (ILP) on the candidate table. CPU only, seconds.

Two analyses on the same table, both reported:
  datatype_only   options with protection p = none only (the 5 datatypes)
  joint           options datatype x protection: p in {none, scale_tmr} (scale_tmr = INT scale word stored 3x, majority vote; implemented and measured)
Reads jfamp_candidate_table.csv (stage5_candidate_table.py) and writes
  normalization_lambda_sweep.csv   every solved setting: mode, normalisation, lambda, constraints, status, objective, storage, predicted risks, assignment
  distinct_assignments.csv         the distinct optimal assignments found in the sweep with how often each occurs
  assignment.json                  the PREDICTED selection at the agreed operating point (layer-additive; the final assignment.json is written by
                                   stage5_end_to_end.py --finalize after the end-to-end measurement)
  optimization_report.md

Objective  min  sum_{l,o} x[l,o] (Q_norm[l,o] + lambda R_norm[l,o])        o = (datatype d, protection p), R = declared-distribution exposure risk (R_exposure_k8_pp)
subject to  one option per layer;  sum S <= B;  sum_l R <= tau_fault (predicted mean accuracy drop under the declared weight-fault distribution);
            joint mode: options with scale-word risk R_scale(l,o) > tau_scale are inadmissible (an unprotected INT scale word is a single point of failure);  x binary.
Clean accuracy A_clean(x) >= A_FP32 - delta_clean and the measured R_fault(x) are checked on the assembled model (stage5_end_to_end.py), not inside the ILP.
Solver: scipy.optimize.milp (HiGHS), exact. Normalisation (global over all candidates): "minmax" -> (v - min)/(max - min), "rank" -> average rank / (n - 1).

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
from fr import protect as P
from fr import stage5 as S5
from fr.config import add_config_args, config_from_args
from fr.utils import dump_json

SHORT = {"e4m11_sn": "E4M11", "fp16": "FP16", "fp8_e4m3fn": "FP8", "int8_pt": "INT8", "int4_pt": "INT4", "int8_pt+scale_tmr": "INT8+T", "int4_pt+scale_tmr": "INT4+T"}
LONG = {v: k for k, v in SHORT.items()}
MODES = ("datatype_only", "joint")


def load_problem(path, q_column="Q_fisher_val_exact", r_column="R_exposure_k8_pp", mode="joint", scale_column="R_scale_pooled_format_pp"):
    rows = list(csv.DictReader(open(path, encoding="utf8")))
    layers = []
    for r in rows:
        if r["layer_name"] not in layers:
            layers.append(r["layer_name"])
    present = {r.get("option", r["datatype"]) for r in rows}
    options = [o for o in S5.OPTIONS if o in present]
    if mode == "datatype_only":
        options = [o for o in options if "+" not in o]
    elif mode != "joint":
        raise SystemExit(f"unknown mode {mode!r}")
    L, D = len(layers), len(options)
    arr = {k: np.zeros((L, D)) for k in ("Q", "R", "S", "Rs", "n")}
    seen = set()
    for r in rows:
        o = r.get("option", r["datatype"])
        if o not in options:
            continue
        i, j = layers.index(r["layer_name"]), options.index(o)
        if (i, j) in seen:
            raise SystemExit("duplicate candidate row")
        seen.add((i, j))
        arr["Q"][i, j] = float(r[q_column])
        arr["R"][i, j] = float(r[r_column])
        arr["S"][i, j] = float(r["S_total_bytes"])                       # includes scales and protection overhead
        arr["Rs"][i, j] = float(r[scale_column]) if r.get(scale_column, "") != "" else 0.0
        arr["n"][i, j] = float(r["num_weights"])
    if len(seen) != L * D:
        raise SystemExit("candidate table is not a full layer x option grid")
    p = S5.Problem(layers, options, arr["Q"], arr["R"], arr["S"], Rscale=arr["Rs"])
    return p, arr["n"][:, 0], rows


def fnum(x, n=6):
    return "" if x is None else f"{x:.{n}g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--table", default=None)
    ap.add_argument("--q-column", default="Q_fisher_val_exact", help="Q_fisher_val_exact (primary) or Q_fisher_train_sampled (labelled alternative)")
    ap.add_argument("--r-column", default="R_exposure_k8_pp", help="R_exposure_k8_pp (primary, declared distribution) or R_any_k8_mean_drop_pp (round 1, unweighted single-layer k = 8)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1, 2, 4, 10])
    ap.add_argument("--avg-bits", type=float, nargs="+", default=[4.5, 5, 6, 7, 8, 10, 12, 16], help="storage budget B as average stored bits per weight")
    ap.add_argument("--tau", type=float, nargs="*", default=[2, 1, 0.5, 0.25], help="tau_fault: max predicted mean accuracy drop (pp) under the declared distribution (a run without it is always added)")
    ap.add_argument("--tau-scale", type=float, default=1.0, help="joint mode: options with scale-word risk above this (pp per flipped scale word) are inadmissible")
    ap.add_argument("--norms", nargs="+", default=["minmax", "rank"])
    ap.add_argument("--select-mode", default="joint", choices=MODES)
    ap.add_argument("--select-norm", default="minmax")
    ap.add_argument("--select-lambda", type=float, default=1.0)
    ap.add_argument("--select-bits", type=float, default=8.0)
    ap.add_argument("--select-tau", type=float, default=0.5)
    args = ap.parse_args()
    cfg = config_from_args(args)
    base = cfg.results_dir() / "jfamp" / "optimization"
    table = Path(args.table) if args.table else base / "jfamp_candidate_table.csv"
    out = Path(args.out_dir) if args.out_dir else table.parent
    out.mkdir(parents=True, exist_ok=True)
    meta_path = table.parent / "candidate_table_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    const_bytes = float(meta.get("fixed_non_quantized_bytes", 0.0))

    problems = {m: load_problem(table, args.q_column, args.r_column, m)[:2] for m in args.modes}
    N = float(next(iter(problems.values()))[1].sum())
    for m, (p, _) in problems.items():
        print(f"{m}: {len(p.layers)} layers x {len(p.formats)} options = {len(p.layers) * len(p.formats)} candidates, {int(N):,} quantized weights")

    taus = [None] + list(args.tau)
    rows, n_solved = [], 0
    count_keys = [(f"n_{SHORT[o].lower().replace('+t', '_tmr')}", o) for o in S5.OPTIONS]
    for mode, (p, nw) in problems.items():
        L = len(p.layers)
        tau_s = args.tau_scale if mode == "joint" else None
        for norm in args.norms:
            for lam in args.lambdas:
                for bits in args.avg_bits:
                    for tau in taus:
                        B = bits * N / 8.0
                        sol = S5.solve(p, lam, norm, "weights", max_bytes=B, max_sum_R=tau, tau_scale=tau_s)
                        row = {"mode": mode, "normalization": norm, "lambda": lam, "B_avg_bits_per_weight": bits, "B_bytes": B, "tau_fault_pred_drop_pp": "" if tau is None else tau,
                               "tau_scale_pp": "" if tau_s is None else tau_s, "status": sol["status"], "objective": "", "storage_bytes": "", "avg_bits_per_weight": "",
                               "model_storage_bytes_incl_fixed": "", "sum_Q": "", "pred_weight_fault_drop_pp": "", "pred_max_scale_word_risk_pp": "", "n_protected_layers": "",
                               **{k: "" for k, _ in count_keys}, "assignment": ""}
                        if sol["choice"] is not None:
                            t = p.totals(sol["choice"])
                            assert t["storage_bytes"] <= B + 1e-6
                            if tau is not None:
                                assert t["sum_R_pp"] <= tau + 1e-9
                            if tau_s is not None:
                                assert t["max_layer_scale_risk_pp"] <= tau_s + 1e-9
                            cnt = Counter(p.formats[c] for c in sol["choice"])
                            row.update({"objective": fnum(sol["objective"], 10), "storage_bytes": int(t["storage_bytes"]), "avg_bits_per_weight": fnum(8 * t["storage_bytes"] / N),
                                        "model_storage_bytes_incl_fixed": int(t["storage_bytes"] + const_bytes), "sum_Q": fnum(t["sum_Q"], 8),
                                        "pred_weight_fault_drop_pp": fnum(t["sum_R_pp"]), "pred_max_scale_word_risk_pp": fnum(t["max_layer_scale_risk_pp"]),
                                        "n_protected_layers": sum(v for o, v in cnt.items() if "+" in o), **{k: cnt.get(o, 0) for k, o in count_keys},
                                        "assignment": ";".join(SHORT[p.formats[c]] for c in sol["choice"])})
                        rows.append(row)
                        n_solved += 1
    with open(out / "normalization_lambda_sweep.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["status"] == "optimal"]
    print(f"{n_solved} problems solved: {len(ok)} optimal, {n_solved - len(ok)} infeasible")

    # ---- distinct assignments (evaluated with the joint problem, which contains every option)
    pj, nwj = problems.get("joint", problems[args.modes[0]])
    distinct = Counter(r["assignment"] for r in ok)
    modes_of = {}
    for r in ok:
        modes_of.setdefault(r["assignment"], set()).add(r["mode"])
    with open(out / "distinct_assignments.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["assignment_id", "found_in_modes", "n_settings", "storage_bytes", "avg_bits_per_weight", "sum_Q", "pred_weight_fault_drop_pp", "pred_max_scale_word_risk_pp",
                    "n_protected_layers", "assignment"])
        for i, (a, c) in enumerate(distinct.most_common()):
            ch = [pj.formats.index(LONG[s]) for s in a.split(";")]
            t = pj.totals(ch)
            w.writerow([f"A{i:03d}", "|".join(sorted(modes_of[a])), c, int(t["storage_bytes"]), fnum(8 * t["storage_bytes"] / N), fnum(t["sum_Q"], 8), fnum(t["sum_R_pp"]),
                        fnum(t["max_layer_scale_risk_pp"]), sum(1 for s in a.split(";") if "+" in s), a])
    print(f"{len(distinct)} distinct optimal assignments")

    # ---- the PREDICTED selection at the agreed operating point
    p, nw = problems[args.select_mode]
    L = len(p.layers)
    B = args.select_bits * N / 8.0
    tau_s = args.tau_scale if args.select_mode == "joint" else None
    sol = S5.solve(p, args.select_lambda, args.select_norm, "weights", max_bytes=B, max_sum_R=args.select_tau, tau_scale=tau_s)
    if sol["choice"] is None:
        sys.exit(f"the selected operating point is {sol['status']}")
    t = p.totals(sol["choice"])
    layers = []
    for l, c in enumerate(sol["choice"]):
        fmt, prot = P.parse_option(p.formats[c])
        layers.append({"layer_id": l, "layer_name": p.layers[l], "num_weights": int(nw[l]), "datatype": fmt, "protection": prot, "Q": p.Q[l, c], "R_exposure_pp": p.R[l, c],
                       "R_scale_word_pp": p.Rscale[l, c], "S_bytes": p.S[l, c]})
    dump_json({"label": ("PREDICTED selection (layer-additive estimates) at the agreed operating point, " + ("joint datatype-protection analysis" if args.select_mode == "joint" else "datatype-only analysis")
                         + ". Not yet measured end to end: the final assignment.json is written by stage5_end_to_end.py --finalize."),
               "mode": args.select_mode, "selected": layers,
               "constraints": {"B_avg_bits_per_weight": args.select_bits, "B_bytes": B, "tau_fault_pred_weight_fault_drop_pp": args.select_tau, "tau_scale_pp": tau_s,
                               "delta_clean_pp": "0.5 (checked end to end after assembly, not an optimiser constraint)"},
               "objective": {"formula": "sum x[l,o] (Q_norm + lambda R_norm)", "q_column": args.q_column, "r_column": args.r_column, "lambda": args.select_lambda,
                             "normalization": args.select_norm, "value": sol["objective"]},
               "totals_predicted": t, "model_storage_bytes_incl_fixed": t["storage_bytes"] + const_bytes,
               "solver": {"name": "scipy.optimize.milp (HiGHS), exact", "scipy": scipy.__version__, "variables": L * len(p.formats)},
               "R_definition": S5.R_DEFINITION, "Q_definition": S5.Q_DEFINITION, "candidate_table_inputs": meta.get("inputs", {})}, out / "assignment.json")

    lines = ["# Stage 5 optimisation", "", f"{n_solved} problems solved ({len(ok)} optimal), {len(distinct)} distinct optimal assignments; full sweep in `normalization_lambda_sweep.csv`.", "",
             f"## Predicted selection at the agreed operating point ({args.select_mode})", "",
             f"B = {args.select_bits} avg bits/weight, lambda = {args.select_lambda}, normalisation {args.select_norm}, tau_fault = {args.select_tau} pp (predicted), tau_scale = {tau_s}.", "",
             "| layer | datatype | protection | Q | R exposure (pp) | scale-word risk (pp) | bytes |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {x['layer_name']} | {S5.LABEL[x['datatype']]} | {x['protection']} | {x['Q']:.3g} | {x['R_exposure_pp']:.3f} | {x['R_scale_word_pp']:.2f} | {int(x['S_bytes']):,} |" for x in layers]
    lines += ["", f"Totals (layer-additive prediction): storage {int(t['storage_bytes']):,} bytes ({8 * t['storage_bytes'] / N:.2f} bits/weight), predicted weight-fault drop {t['sum_R_pp']:.3f} pp, "
              f"largest scale-word risk {t['max_layer_scale_risk_pp']:.2f} pp. Not measured end to end yet."]
    open(out / "optimization_report.md", "w", encoding="utf8").write("\n".join(lines) + "\n")
    print(f"selected ({args.select_mode}): {t['storage_bytes']:,.0f} bytes, predicted weight-fault drop {t['sum_R_pp']:.3f} pp, max scale risk {t['max_layer_scale_risk_pp']:.2f} pp -> {out / 'assignment.json'}")


if __name__ == "__main__":
    main()
