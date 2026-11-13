"""Stage 6 helpers: Pareto front, Spearman alignment table, markdown table; no data files needed."""
import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("stage6", Path(__file__).resolve().parents[1] / "scripts" / "stage6_final_package.py")
S6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S6)


def test_pareto_keeps_only_non_dominated_rows():
    pts = [[4, 1.2], [5, 0.5], [5, 0.6], [8, 0.1], [9, 0.1], [16, 0.0]]
    assert S6.pareto(pts) == [0, 1, 3, 5]


def test_pareto_ties_are_both_kept():
    assert S6.pareto([[1, 1], [1, 1]]) == [0, 1]


def test_alignment_table_detects_perfect_monotone_relation():
    rows = []
    for i in range(21):
        rows.append({"format": "fp16", "layer_id": i, "layer_name": f"l{i}", "num_weights": 100 + i, "weight_std": 1.0 / (i + 1), "weight_abs_max": 1.0, "sqnr_db": float(i % 3),
                     "risk_k1_mean_drop_pp": float(i), "risk_k8_mean_drop_pp": float(i)})
    rows = rows + [dict(r, format=f) for f in S6.FORMATS if f != "fp16" for r in rows]
    A = S6.table_alignment(rows)
    r = next(x for x in A if x["format"] == "fp16" and x["k"] == "8" and x["predictor"] == "num_weights")
    assert abs(r["spearman_rho"] - 1.0) < 1e-12 and r["significant_5pct"]
    r = next(x for x in A if x["format"] == "fp16" and x["k"] == "8" and x["predictor"] == "weight_std")
    assert abs(r["spearman_rho"] + 1.0) < 1e-12


def test_md_table_shape():
    md = S6.md_table([{"a": 1, "b": 2.5}], ["a", "b"], ["A", "B"], {"b": lambda v: f"{v:.1f}"})
    assert md.splitlines()[0] == "| A | B |" and md.splitlines()[2] == "| 1 | 2.5 |"


def test_float_helper_and_nan():
    assert S6.fl("1.5") == 1.5 and np.isnan(S6.fl("")) and S6.fl(None, 0.0) == 0.0
