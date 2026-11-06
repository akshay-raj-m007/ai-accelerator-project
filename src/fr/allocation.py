"""Layer scores and the choice of which layers move from E4M11 (16 bit) to an 8-bit format.

Per layer i (Conv2d/Linear weight w_i, 8-bit reconstruction w^_i, Fisher diagonal F_i):

    error_i   = sum (w - w^)^2 / sum w^2                       relative weight error of the 8-bit format
    fisher_i  = sum F / n_i  (``per_param``)  or  sum F  (``total``)      Fisher trace (LAMPQ metric)
    damage_i  = sum F * (w - w^)^2                              second-order loss estimate of the switch
                                                                (additive over layers; the principled form)

Scorers (higher = more sensitive = keep as E4M11):

    ``lambda``   S_i = lam * norm(error_i) + (1 - lam) * norm(fisher_i)       (the requested weighted sum)
                 lam = 0 is Fisher only (independent of the 8-bit format), lam = 1 is error only
    ``product``  S_i = damage_i
    ``random``   uniformly random feasible set (control)
    ``worst``    the feasible set with the *largest* total damage (sanity envelope)

Selection (``select``): a size target (average bits per weight, 16 for E4M11 and 8 for the 8-bit format)
fixes the number of weights that must be switched, ``need = N (16 - T) / 8``. All 2^L subsets of layers
(L = 21, 2M subsets) are enumerated and those switching between ``need`` and ``need (1 + tol)`` weights are
feasible (``tol`` is doubled until one exists). The chosen set minimises the summed score. This is the
LAMPQ-style ILP solved exactly; random and worst use the identical feasible set so budgets are matched.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from . import formats8

NORMS = ("rank", "zscore", "minmax")
FISHER_MODES = ("per_param", "total")
HI_BITS, LO_BITS = 16, 8
MAX_LAYERS = 24


def layer_table(weights: Dict[str, np.ndarray], fisher: Dict[str, np.ndarray], fmt: str,
                cache: dict | None = None) -> List[dict]:
    """One row per layer for the 8-bit format ``fmt`` (``cache`` memoises ``quantize8`` results)."""
    if list(weights) != list(fisher):
        raise ValueError("weights and fisher must have the same layers in the same order")
    rows = []
    for name, w in weights.items():
        key = (name, fmt)
        if cache is not None and key in cache:
            q = cache[key]
        else:
            q = formats8.quantize8(w, fmt)
            if cache is not None:
                cache[key] = q
        w64 = w.astype(np.float64).reshape(-1)
        d = w64 - q["dequantized"].astype(np.float64).reshape(-1)
        f = np.asarray(fisher[name], dtype=np.float64).reshape(-1)
        sig, err = float(np.dot(w64, w64)), float(np.dot(d, d))
        rows.append({
            "name": name, "count": int(w64.size),
            "error": err / sig if sig > 0 else 0.0,
            "fisher_total": float(f.sum()), "fisher_per_param": float(f.sum() / w64.size),
            "damage": float(np.dot(f, d * d)),
        })
    return rows


def normalize(values: Sequence[float], how: str = "rank") -> np.ndarray:
    """Map scores to a comparable scale: ``rank`` (average ranks, scaled to [0, 1]), ``zscore``, ``minmax``."""
    v = np.asarray(values, dtype=np.float64)
    if how == "rank":
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(len(v), dtype=np.float64)
        ranks[order] = np.arange(len(v), dtype=np.float64)
        for val in np.unique(v):                        # average ranks over ties
            m = v == val
            ranks[m] = ranks[m].mean()
        return ranks / max(len(v) - 1, 1)
    if how == "zscore":
        sd = v.std()
        return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
    if how == "minmax":
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 0 else np.zeros_like(v)
    raise ValueError(f"norm must be one of {NORMS}")


def combined_score(rows: List[dict], lam: float, norm: str = "rank",
                   fisher_mode: str = "per_param") -> np.ndarray:
    """S_i = lam * norm(error_i) + (1 - lam) * norm(fisher_i)."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    if fisher_mode not in FISHER_MODES:
        raise ValueError(f"fisher_mode must be one of {FISHER_MODES}")
    err = normalize([r["error"] for r in rows], norm)
    fis = normalize([r["fisher_per_param" if fisher_mode == "per_param" else "fisher_total"]
                     for r in rows], norm)
    return lam * err + (1.0 - lam) * fis


def required_switched(total: int, target_avg_bits: float) -> int:
    """Weights that must be 8-bit so that the payload average is <= ``target_avg_bits``."""
    if not LO_BITS <= target_avg_bits <= HI_BITS:
        raise ValueError(f"target average bits must be in [{LO_BITS}, {HI_BITS}]")
    return int(np.ceil(total * (HI_BITS - target_avg_bits) / (HI_BITS - LO_BITS) - 1e-9))


def _enumerate(sizes: Sequence[int], costs: Sequence[float]):
    if len(sizes) > MAX_LAYERS:
        raise ValueError(f"exact enumeration supports at most {MAX_LAYERS} layers")
    s = np.zeros(1, dtype=np.int64)
    c = np.zeros(1, dtype=np.float64)
    for size, cost in zip(sizes, costs):                # bit i of the subset index = layer i switched
        s = np.concatenate([s, s + int(size)])
        c = np.concatenate([c, c + float(cost)])
    return s, c


def feasible_subsets(sizes: Sequence[int], need: int, tol: float = 0.05):
    """Indices of subsets whose switched-weight total lies in [need, need (1 + tol)]; ``tol`` grows if empty.

    Returns (indices, sums, tol_used)."""
    s, _ = _enumerate(sizes, np.zeros(len(sizes)))
    if need > s[-1]:
        raise ValueError("size target is unreachable even with every layer at 8 bit")
    t = tol
    while True:
        idx = np.nonzero((s >= need) & (s <= need * (1.0 + t) + 1))[0]
        if idx.size or t > 4.0:
            return idx, s, t
        t = t * 2 if t > 0 else 0.05


def mask_to_names(mask: int, names: Sequence[str]) -> List[str]:
    return [n for i, n in enumerate(names) if (mask >> i) & 1]


def select(method: str, names: Sequence[str], sizes: Sequence[int], target_avg_bits: float,
           costs: Sequence[float] | None = None, damages: Sequence[float] | None = None,
           tol: float = 0.05, seed: int = 0) -> dict:
    """Choose the layers to switch to 8 bit.

    method: ``min_cost`` (minimise sum of ``costs`` over the feasible sets), ``random`` (uniform over the
    feasible sets, ``seed``), ``worst`` (maximise sum of ``damages``). Returns a dict with ``layers8``,
    ``n_switched``, ``need``, ``tol_used``, ``total_cost`` and the achieved ``avg_bits`` (payload only).
    """
    sizes = [int(x) for x in sizes]
    total = sum(sizes)
    need = required_switched(total, target_avg_bits)
    idx, sums, tol_used = feasible_subsets(sizes, need, tol)
    if idx.size == 0:
        raise ValueError("no feasible layer subset for this target")
    if method == "min_cost":
        if costs is None:
            raise ValueError("costs required")
        _, c = _enumerate(sizes, costs)
        pick = int(idx[np.argmin(c[idx])])
        cost = float(c[pick])
    elif method == "random":
        rng = np.random.default_rng(seed)
        pick = int(rng.choice(idx))
        cost = None
    elif method == "worst":
        if damages is None:
            raise ValueError("damages required")
        _, c = _enumerate(sizes, damages)
        pick = int(idx[np.argmax(c[idx])])
        cost = float(c[pick])
    else:
        raise ValueError("method must be min_cost | random | worst")
    n8 = int(sums[pick])
    return {"layers8": mask_to_names(pick, names), "mask": pick, "n_switched": n8, "need": need,
            "tol_used": tol_used, "n_feasible": int(idx.size), "total_cost": cost,
            "avg_bits": (HI_BITS * (total - n8) + LO_BITS * n8) / total}
