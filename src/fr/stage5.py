"""J-FAMP Stage 5: per-layer candidate table (Q, R, S) and the constrained assignment optimisation (runbook Section 7).

Candidate = (layer l, datatype d, protection p). Implemented and measured protection options: ``none`` and ``scale_tmr`` (INT layers only: the float32 scale word is
stored three times and read by bit-wise majority, see protect.py). The DATATYPE-ONLY analysis uses p = none only; the JOINT analysis uses both options.

    Q(l,d)  Fisher-weighted reconstruction penalty   Q = sum_i F_i (w_i - w^_i)^2      (second-order loss estimate; w^ = decoded stored weights of the
            frozen Stage 2 encoding). PRIMARY: F = exact-expectation diagonal Fisher on held-out validation images 0..2047. ALTERNATIVE (labelled): sampled-label
            true Fisher on 1,024 training images. 0.5 * Q is the predicted clean cross-entropy increase if only this layer is stored in d.
    R(l,d,p) PRIMARY residual fault risk (declared distribution: random flips over all stored weight bits of the assembled model, 8 distinct weights per trial):
            R_exposure = 8 * (n_l / N) * max(0, per-flip drop of layer l),  n_l = weights of the layer, N = all quantizable weights, per-flip drop = mean of the
            two Stage 4 estimates (k = 1 mean drop, and k = 8 mean drop / 8; any bit of the stored weight, 20 trials x 1,000 validation images each).
            It is additive over layers and in the unit of the end-to-end metric, so sum_l R_exposure predicts the measured mean accuracy drop of the assembled model
            under the declared distribution. The scale-word fault (one flipped float32 scale bit) is a SECOND metric (R_scale, never merged into R): unprotected INT
            layers carry the Stage 4 value, scale_tmr layers carry the value measured on the protected store (stage5_tmr_layerwise.py).
            Round 1 used the unweighted mean over layers of the single-layer k = 8 drop (column R_any_k8_mean_drop_pp), which did not match the declared distribution.
    S(l,d,p) actual stored bytes of the layer: payload + float32 scale(s) (INT) + protection overhead (scale_tmr: 2 extra copies = 8 bytes) + padding (none occurs).

Layer-additive scores are an approximation. The selected assignment must be assembled and measured end to end (stage5 end-to-end script).
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

FORMATS = ["e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt"]
LABEL = {"e4m11_sn": "E4M11-SN", "fp16": "FP16", "fp8_e4m3fn": "FP8 E4M3FN", "int8_pt": "INT8 per-tensor", "int4_pt": "INT4 per-tensor"}
PROTECTION = "none"
OPTIONS = ["e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt", "int8_pt+scale_tmr", "int4_pt+scale_tmr"]      # datatype[+protection] tokens (protect.py)
DATATYPE_ONLY_OPTIONS = FORMATS
Q_DEFINITION = ("PRIMARY Q (Q_fisher_val_exact) = sum_i F_i (w_i - w_hat_i)^2 with F = diagonal Fisher of the FP32 model, expectation over labels computed EXACTLY, on held-out "
                "validation images 0..2047; w_hat = decoded frozen Stage 2 encoding. ALTERNATIVE (Q_fisher_train_sampled): same with the sampled-label true Fisher on 1,024 training images "
                "(this network is saturated on its training images and the sampled estimator undercounts; see phase5/Q_ANALYSIS.md)")
R_DEFINITION = ("R_exposure(l,d,p) = 8 * (n_l/N) * max(0, mean(k1 drop, k8 drop / 8)) in percentage points, Stage 4 'any bit' layer-wise drops (20 trials x 1,000 validation images "
                "each); sum over layers = predicted mean accuracy drop under 8 random flips over all stored weight bits of the assembled model. Scale-word faults are a separate metric.")
LABEL.update({"int8_pt+scale_tmr": "INT8 + scale TMR", "int4_pt+scale_tmr": "INT4 + scale TMR"})


# ------------------------------------------------------------------------------------------------ building the table
def fisher_q(w: np.ndarray, w_hat: np.ndarray, fisher: np.ndarray) -> float:
    d = np.asarray(w, dtype=np.float64).reshape(-1) - np.asarray(w_hat, dtype=np.float64).reshape(-1)
    f = np.asarray(fisher, dtype=np.float64).reshape(-1)
    if f.shape != d.shape:
        raise ValueError("fisher and weights differ in size")
    return float(np.dot(f, d * d))


def risk_lookup(fault_rows: Sequence[dict]) -> Dict[tuple, dict]:
    """(format, layer_name, scope_kind, group, k) -> summary row of Stage 4 fault_summary.csv (layer-wise rows only)."""
    out = {}
    for r in fault_rows:
        out[(r["format"], r["layer_name"], r["scope_kind"], r["fault_group"], int(r["k"]))] = r
    return out


def risk_columns(risk: Dict[tuple, dict], fmt: str, layer: str) -> dict:
    def get(scope, group, k, col, default=None):
        r = risk.get((fmt, layer, scope, group, k))
        return float(r[col]) if r is not None else default

    main = risk.get((fmt, layer, "weights", "all", 8))
    if main is None:
        raise KeyError(f"no Stage 4 row for {fmt}/{layer}/weights/all/k8")
    scale = risk.get((fmt, layer, "scales", "all", 1))
    return {
        "R_any_k8_mean_drop_pp": float(main["drop_pp_mean"]), "R_any_k8_p90_drop_pp": float(main["drop_pp_p90"]),
        "R_any_k8_worst_drop_pp": float(main["drop_pp_worst"]), "R_any_k8_failure_rate": float(main["failure_rate"]),
        "R_any_k1_mean_drop_pp": get("weights", "all", 1, "drop_pp_mean"),
        "R_exponent_k8_mean_drop_pp": get("weights", "exponent", 8, "drop_pp_mean", ""),
        "R_scale_any_k1_mean_drop_pp": float(scale["drop_pp_mean"]) if scale is not None else 0.0,
        "has_scale_word": scale is not None,
        "R_n_trials": int(main["n_trials"]), "R_sample_count": int(main["sample_count"])}


def per_flip_drop(k1_drop: float, k8_drop: float) -> float:
    """Per-flip accuracy drop (pp) of a layer: mean of the k = 1 estimate and the k = 8 estimate / 8, clipped at 0 (negative values are noise)."""
    return max(0.0, 0.5 * (float(k1_drop) + float(k8_drop) / 8.0))


def exposure_risk(n_layer: int, n_total: int, k1_drop: float, k8_drop: float, k: int = 8) -> float:
    """Declared-distribution risk of a layer: k flips spread over all weights land in this layer with expectation k * n_l / N."""
    return k * (n_layer / n_total) * per_flip_drop(k1_drop, k8_drop)


# ------------------------------------------------------------------------------------------------ normalisation
def normalize(values: Sequence[float], how: str = "minmax") -> np.ndarray:
    """Global normalisation over ALL candidates (so Q and R are on [0, 1] before the lambda weighting)."""
    v = np.asarray(values, dtype=np.float64)
    if how == "minmax":
        lo, hi = float(v.min()), float(v.max())
        return np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo)
    if how == "rank":
        order = v.argsort(kind="mergesort")
        ranks = np.empty(len(v))
        sv = v[order]
        i = 0
        while i < len(v):                                   # average rank for ties
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return ranks / max(len(v) - 1, 1)
    raise ValueError(how)


# ------------------------------------------------------------------------------------------------ optimisation
class Problem:
    """Arrays over (layer, option) candidates, layer-major. ``formats`` are option labels (datatype or datatype+protection tokens)."""

    def __init__(self, layers: Sequence[str], formats: Sequence[str], Q, R, S, Rscale=None, dce=None, allowed=None):
        self.layers, self.formats = list(layers), list(formats)
        L, D = len(layers), len(formats)
        self.Q, self.R, self.S = (np.asarray(a, dtype=np.float64).reshape(L, D) for a in (Q, R, S))
        self.Rscale = np.zeros((L, D)) if Rscale is None else np.asarray(Rscale, dtype=np.float64).reshape(L, D)
        self.dce = 0.5 * self.Q if dce is None else np.asarray(dce, dtype=np.float64).reshape(L, D)
        self.allowed = np.ones((L, D), dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool).reshape(L, D)

    def totals(self, choice: Sequence[int]) -> dict:
        idx = np.arange(len(self.layers))
        c = np.asarray(choice)
        return {"storage_bytes": float(self.S[idx, c].sum()), "sum_Q": float(self.Q[idx, c].sum()),
                "mean_R_pp": float(self.R[idx, c].mean()), "sum_R_pp": float(self.R[idx, c].sum()),
                "mean_R_plus_scale_pp": float((self.R + self.Rscale)[idx, c].mean()),
                "max_layer_scale_risk_pp": float(self.Rscale[idx, c].max()), "predicted_dCE": float(self.dce[idx, c].sum())}


def solve(p: Problem, lam: float, norm: str = "minmax", risk: str = "weights", max_bytes: Optional[float] = None,
          max_mean_R: Optional[float] = None, max_dce: Optional[float] = None, max_sum_R: Optional[float] = None, tau_scale: Optional[float] = None) -> dict:
    """min sum x[l,d] (Qn + lam Rn)  s.t. one option per layer, sum S <= max_bytes, mean R <= max_mean_R, sum R <= max_sum_R (declared-distribution risk),
    sum 0.5 Q <= max_dce, x binary; with ``tau_scale`` only options whose scale-word risk R_scale(l,o) <= tau_scale are admissible (and ``p.allowed``)."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    L, D = len(p.layers), len(p.formats)
    Rv = p.R if risk == "weights" else p.R + p.Rscale          # "plus_scale": union of the two fault processes (sensitivity variant)
    qn = normalize(p.Q.reshape(-1), norm)
    rn = normalize(Rv.reshape(-1), norm)
    c = qn + lam * rn
    allowed = p.allowed.copy()
    if tau_scale is not None:
        allowed &= p.Rscale <= tau_scale + 1e-12
    A, lo, hi = [], [], []
    for l in range(L):
        row = np.zeros(L * D)
        row[l * D:(l + 1) * D] = 1
        A.append(row), lo.append(1), hi.append(1)
    if max_bytes is not None:
        A.append(p.S.reshape(-1)), lo.append(-np.inf), hi.append(max_bytes)
    if max_mean_R is not None:
        A.append(Rv.reshape(-1) / L), lo.append(-np.inf), hi.append(max_mean_R)
    if max_sum_R is not None:
        A.append(Rv.reshape(-1)), lo.append(-np.inf), hi.append(max_sum_R)
    if max_dce is not None:
        A.append(p.dce.reshape(-1)), lo.append(-np.inf), hi.append(max_dce)
    ub = allowed.reshape(-1).astype(float)
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(L * D), bounds=Bounds(np.zeros(L * D), ub))
    if not res.success:
        return {"status": "infeasible" if res.status == 2 else f"failed({res.status})", "choice": None, "objective": None}
    x = np.round(res.x).reshape(L, D)
    choice = [int(np.argmax(x[l])) for l in range(L)]
    return {"status": "optimal", "choice": choice, "objective": float(res.fun)}


def brute_force(p: Problem, lam: float, norm: str = "minmax", risk: str = "weights", max_bytes=None, max_mean_R=None, max_dce=None, max_sum_R=None, tau_scale=None):
    """Exact reference for tiny problems (tests)."""
    L, D = len(p.layers), len(p.formats)
    Rv = p.R if risk == "weights" else p.R + p.Rscale
    qn = normalize(p.Q.reshape(-1), norm).reshape(L, D)
    rn = normalize(Rv.reshape(-1), norm).reshape(L, D)
    allowed = p.allowed.copy()
    if tau_scale is not None:
        allowed &= p.Rscale <= tau_scale + 1e-12
    best = None
    for ch in itertools.product(range(D), repeat=L):
        idx = np.arange(L)
        if not allowed[idx, ch].all():
            continue
        if max_bytes is not None and p.S[idx, ch].sum() > max_bytes + 1e-9:
            continue
        if max_mean_R is not None and Rv[idx, ch].mean() > max_mean_R + 1e-9:
            continue
        if max_sum_R is not None and Rv[idx, ch].sum() > max_sum_R + 1e-9:
            continue
        if max_dce is not None and p.dce[idx, ch].sum() > max_dce + 1e-12:
            continue
        obj = (qn + lam * rn)[idx, ch].sum()
        if best is None or obj < best[0] - 1e-12:
            best = (float(obj), list(ch))
    return best
