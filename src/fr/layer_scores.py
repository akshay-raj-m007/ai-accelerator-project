"""Plan B: per-layer scores that say how dangerous it is to keep a layer in E4M11 under random bit flips.

All scores are computed on the all-E4M11 model (the paper's format) and use held-out data only (see the scripts).

  S0   clean Fisher trace of the layer, tr(F_i) and tr(F_i)/N_i (LAMPQ sensitivity, no fault model involved)
  S1   MEASURED damage: k random bits are flipped in layer i ONLY (all other layers fault-free); damage is the increase
       of the cross-entropy over the clean model, the KL divergence of the output distribution, and the accuracy drop,
       averaged over trials (common random numbers: trial t uses the same RNG stream for every score of that layer)
  S2   PREDICTED damage from the Fisher and the fault model. LAMPQ Lemma 1 / HAWQ-V2: for a small weight perturbation
       dw the loss rises by about 1/2 * sum_j F_jj dw_j^2. A single flip of a uniformly chosen weight and a uniformly
       chosen bit of `group` therefore costs, in expectation,
           S2_i = 1/2 * (1/N_i) * sum_j F_jj * mean_b (w'_jb - w_j)^2        (b over the bits of `group`)
       where w'_jb is the E4M11 value after flipping bit b of weight j (exact decode, no approximation of dw).
  S2n  the same without the Fisher, 1/2 * (1/N_i) * sum_j mean_b (w'_jb - w_j)^2: the expected squared perturbation.
       Comparing S2 with S2n tells whether the Fisher adds anything beyond "how big is the damage in weight space".

S1 and S2 are per layer; to get the damage of a layer under a model-wide load of K flips multiply a per-flip score by
K * N_i / N (bit-uniform load: the layer is struck in proportion to its size).
Nothing here is tuned; the second-order formula is a prediction that S1 can contradict (it is exactly what the rank
correlation is meant to test: large exponent flips are far outside the small-perturbation regime of Lemma 1).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import faults_mixed as FM
from .faults import trial_rng


def rank(x: Sequence[float]) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size)
    r[order] = np.arange(x.size, dtype=np.float64)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def predicted_flip_moments(inj: "FM.MixedFaultInjector", fisher: Dict[str, torch.Tensor], group: str) -> List[dict]:
    """S2 and S2n for every layer (exact decode of every single-bit flip of every weight)."""
    out = []
    for l, name in enumerate(inj.names):
        bits = FM.group_bits(inj.fmts[l], group)
        if not bits:
            out.append({"layer": name, "count": int(inj.counts[l]), "S2": 0.0, "S2n": 0.0, "n_bits": 0,
                        "max_abs_delta": 0.0})
            continue
        w = inj._decode(l, inj.words[l], np.arange(inj.counts[l])).astype(np.float64)
        f = fisher[name].reshape(-1).double().numpy()
        if f.size != w.size:
            raise ValueError(f"Fisher of {name} has {f.size} entries, the buffer {w.size}")
        sq = np.zeros_like(w)
        mx = 0.0
        for b in bits:
            flipped = inj.words[l] ^ np.asarray(1 << b, dtype=inj.words[l].dtype)
            d = inj._decode(l, flipped, np.arange(inj.counts[l])).astype(np.float64) - w
            sq += d * d
            mx = max(mx, float(np.abs(d).max()))
        sq /= len(bits)                                    # mean over the bits of the group
        n = w.size
        out.append({"layer": name, "count": int(n), "S2": float(0.5 * (f * sq).sum() / n),
                    "S2n": float(0.5 * sq.sum() / n), "n_bits": len(bits), "max_abs_delta": mx})
    return out


def measured_layer_damage(inj: "FM.MixedFaultInjector", X, y, group: str, k: int, trials: int, seed: int,
                          sampling: str = "bits", log=print) -> List[dict]:
    """S1: for every layer, strike k bits of ``group`` in that layer only; mean/median damage over ``trials``."""
    clean_logits = FM.logits_of(inj.model, X)
    clean_ce = torch.nn.functional.cross_entropy(clean_logits, y).item()
    clean_acc = 100.0 * (clean_logits.argmax(1) == y).float().mean().item()
    rows = []
    for l, name in enumerate(inj.names):
        if not FM.group_bits(inj.fmts[l], group):
            continue
        ce, kl, acc, nonfin = [], [], [], 0
        for t in range(trials):
            rng = trial_rng(seed, f"layer|{name}|{group}|{sampling}", k, t)
            idx, b = inj.draw(rng, group, k, sampling, layer=name)
            inj.inject(idx, b)
            try:
                m = FM.trial_metrics(FM.logits_of(inj.model, X), clean_logits, y)
            finally:
                inj.restore()
            if not m["finite"]:
                nonfin += 1
            ce.append(m["cross_entropy"] - clean_ce)
            kl.append(m["kl_vs_clean"])
            acc.append(clean_acc - m["accuracy"])
        ce, kl, acc = (np.asarray(v, dtype=np.float64) for v in (ce, kl, acc))
        rows.append({"layer": name, "count": int(inj.counts[l]), "group": group, "k": k, "trials": trials,
                     "d_ce_mean": float(np.nanmean(ce)), "d_ce_median": float(np.nanmedian(ce)),
                     "kl_mean": float(np.nanmean(kl)), "kl_median": float(np.nanmedian(kl)),
                     "d_acc_mean": float(acc.mean()), "d_acc_median": float(np.median(acc)),
                     "d_acc_p90": float(np.quantile(acc, 0.9)), "non_finite": nonfin})
        log(f"  [{group} k={k}] {name:28s} dCE mean {rows[-1]['d_ce_mean']:+.4f}  dAcc mean "
            f"{rows[-1]['d_acc_mean']:+.2f} pp")
    return rows
