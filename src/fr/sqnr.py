"""SQNR (paper Eq. 6) with per-layer and magnitude-bin breakdowns.

Paper Sec 5.2: "SQNR is defined as SQNR = 10 log10( sum_i w_i^2 / sum_i (w_i - w^_i)^2 ) dB
... measured over convolutional and fully connected weight tensors."
The paper writes one ratio; we therefore pool *all* Conv2d + Linear weights into a
single sum (global SQNR) and additionally report each layer on its own.
Everything is accumulated in float64.

Magnitude bins use the *original* weight: |w| < 2^-7 (flushed to zero by
Algorithm 1 step 3), 2^-7 <= |w| < 2^-6 (rounded up to the minimum normal),
|w| >= 2^-6 (representable normal range).
"""
from __future__ import annotations

import math
from typing import Dict, Mapping

import numpy as np

BIN_EDGES = (2.0 ** -7, 2.0 ** -6)
BIN_NAMES = ("abs_lt_2^-7 (flushed)", "2^-7<=abs<2^-6", "abs>=2^-6")


def sqnr_db(signal_power: float, error_power: float) -> float:
    if error_power == 0.0:
        return math.inf          # lossless storage
    if signal_power == 0.0:
        return -math.inf
    return 10.0 * math.log10(signal_power / error_power)


def _powers(w: np.ndarray, w_hat: np.ndarray):
    w = w.astype(np.float64, copy=False).reshape(-1)
    h = w_hat.astype(np.float64, copy=False).reshape(-1)
    d = w - h
    return float(np.dot(w, w)), float(np.dot(d, d))


def weight_stats(arrs) -> dict:
    """min / max / |w|max / mean / std over one array or an iterable of arrays."""
    if isinstance(arrs, np.ndarray):
        arrs = [arrs]
    n = 0
    s = 0.0
    s2 = 0.0
    lo, hi = math.inf, -math.inf
    for a in arrs:
        a = a.astype(np.float64, copy=False).reshape(-1)
        n += a.size
        s += float(a.sum())
        s2 += float(np.dot(a, a))
        lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    mean = s / n
    var = max(s2 / n - mean * mean, 0.0)
    return {"count": n, "min": lo, "max": hi, "abs_max": max(abs(lo), abs(hi)),
            "mean": mean, "std": math.sqrt(var)}


def sqnr_report(original: Mapping[str, np.ndarray], dequantized: Mapping[str, np.ndarray]) -> dict:
    """Eq. 6 over ``original`` (FP32) vs ``dequantized`` tensors (same keys)."""
    if list(original) != list(dequantized):
        raise ValueError("original and dequantized must have identical keys/order")
    per_layer: Dict[str, dict] = {}
    sig_tot = err_tot = 0.0
    n_tot = 0
    bins = [{"name": nme, "count": 0, "signal_power": 0.0, "error_power": 0.0} for nme in BIN_NAMES]
    for name, w in original.items():
        h = dequantized[name]
        sig, err = _powers(w, h)
        flat = np.abs(w.astype(np.float64).reshape(-1))
        masks = (flat < BIN_EDGES[0], (flat >= BIN_EDGES[0]) & (flat < BIN_EDGES[1]),
                 flat >= BIN_EDGES[1])
        wf = w.astype(np.float64).reshape(-1)
        df = wf - h.astype(np.float64).reshape(-1)
        for b, mk in zip(bins, masks):
            b["count"] += int(mk.sum())
            b["signal_power"] += float(np.dot(wf[mk], wf[mk]))
            b["error_power"] += float(np.dot(df[mk], df[mk]))
        per_layer[name] = {
            "count": int(flat.size), "signal_power": sig, "error_power": err,
            "sqnr_db": sqnr_db(sig, err),
            "frac_abs_lt_2^-6": float(masks[0].mean() + masks[1].mean()),
            "stats": weight_stats(w),
        }
        sig_tot += sig
        err_tot += err
        n_tot += flat.size
    for b in bins:
        b["fraction_of_weights"] = b["count"] / n_tot
        b["fraction_of_signal_power"] = b["signal_power"] / sig_tot if sig_tot else 0.0
        b["fraction_of_error_power"] = b["error_power"] / err_tot if err_tot else 0.0
        b["sqnr_db"] = sqnr_db(b["signal_power"], b["error_power"])
    return {
        "definition": "paper Eq. 6, pooled over all listed tensors (float64)",
        "n_layers": len(per_layer),
        "overall": {"count": n_tot, "signal_power": sig_tot, "error_power": err_tot,
                    "sqnr_db": sqnr_db(sig_tot, err_tot)},
        "breakdown": bins,
        "per_layer": per_layer,
        "stats": weight_stats(list(original.values())),
    }
