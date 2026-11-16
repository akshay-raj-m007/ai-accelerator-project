"""Weight-space SQNR of an assembled (mixed datatype + protection) model, clean and after bit flips in the stored bytes.

    SQNR = 10 log10( sum_l ||w_l||^2 / sum_l ||w_l - w_hat_l||^2 )     over all quantized Conv2d / Linear weights, w = FP32 checkpoint, w_hat = decoded stored weights

"Initial" (FP32) has no quantization noise: error 0, SQNR = +inf (reference). "After quantization" = decoded frozen Stage 2 encodings. "After bit flips" = the decoded weights after
the faults of one trial (flips drawn on the stored bits, passed through each layer's protection, residual flips applied to the stored bytes, exactly as in the end-to-end accuracy
trials); a trial in which a decoded weight is non-finite has SQNR = -inf and is counted separately, never dropped.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import stage3 as S3


def sqnr_db(signal: float, err: float) -> float:
    if not (math.isfinite(err) and math.isfinite(signal)):
        return float("-inf")
    if err <= 0.0:
        return float("inf")
    return 10.0 * math.log10(signal / err)


class SqnrMeter:
    """Tracks the squared weight error of a MixedFaultInjector against the FP32 weights; after an inject() only the touched layers are re-measured."""

    def __init__(self, fp32_weights: Sequence[torch.Tensor], inj):
        self.W = [w.detach().double().reshape(-1).cpu() for w in fp32_weights]
        if len(self.W) != len(inj.flat):
            raise ValueError("layer count differs between the FP32 weights and the injector")
        self.inj = inj
        self.signal = float(sum(float(w.pow(2).sum()) for w in self.W))
        self.err = [self._layer_err(l) for l in range(len(self.W))]
        self.err_total = float(sum(self.err))

    def _layer_err(self, l: int) -> float:
        d = self.inj.flat[l].detach().double().cpu() - self.W[l]
        return float(d.pow(2).sum())

    def clean_db(self) -> float:
        return sqnr_db(self.signal, self.err_total)

    def current_db(self, touched: Sequence[int]):
        """SQNR with the faults currently injected. ``touched`` = layers whose weights may have changed. -> (db, nonfinite)"""
        touched = sorted(set(int(l) for l in touched))
        e = self.err_total
        nonfinite = False
        for l in touched:
            new = self._layer_err(l)
            if not math.isfinite(new):
                nonfinite = True
            e += new - self.err[l]
        if nonfinite:
            return float("-inf"), True
        return sqnr_db(self.signal, e), False


def draw_flips(mode: str, pf, store, dist: str, seed: int, k: int):
    """-> (residual weight flips, residual scale flips) of one trial. mode 's6' = S6.ProtectedFaults path (MobileNetV2 run), 'e2' = Stage 5 draws (ResNet-18 run)."""
    if mode == "s6":
        from . import stage6 as S6
        _, rw, rs = S6.draw_trial_faults(pf, dist, seed, k)
        return list(rw), list(rs)
    if mode == "e2":
        from . import stage5_e2e as E2
        rng = np.random.default_rng(seed)
        if dist == "weights_k8":
            return list(E2.draw_global_weight_flips(rng, store, k)), []
        if dist == "scales_k1":
            return [], list(E2.draw_scale_flips(rng, store, k))
        raise ValueError(dist)
    raise ValueError(mode)


def flips_record(store, wf, sf) -> list:
    """The ``flips`` field of an end-to-end trial record (so the draws can be compared with the recorded trials)."""
    out = [["w", int(l), int(e), int(b), store.fmt_of[l]] for l, e, b in wf] + [["s", int(l), int(c), int(b), store.fmt_of[l]] for l, c, b in sf]
    return json.loads(json.dumps(out))


def run_assignment(meter: SqnrMeter, pf, store, aid: str, dists: Sequence[str], dist_k: Dict[str, int], trials: int, base_seed: int, mode: str,
                   recorded: Optional[Dict[str, dict]] = None) -> List[dict]:
    """SQNR of every fault trial of one assembled model. Restores the stored bytes after every trial and checks the byte-exact restoration."""
    inj = meter.inj
    rows = []
    for dist in dists:
        k = dist_k[dist]
        for t in range(trials):
            seed = S3.trial_seed(base_seed, f"mixed:{aid}", "GLOBAL", dist, "all", k, t)
            wf, sf = draw_flips(mode, pf, store, dist, seed, k)
            h0 = store.hash()
            try:
                inj.inject(wf, sf)
                db, nonfinite = meter.current_db([l for l, *_ in wf] + [x[0] for x in sf])
            finally:
                inj.restore()
            restored = store.hash() == h0
            rec = (recorded or {}).get(f"{aid}|{dist}|k{k}|t{t}")
            same = None if rec is None else (rec["flips"] == flips_record(store, wf, sf))
            drop = None if rec is None else float(rec["clean_accuracy"]) - float(rec["fault_accuracy"])
            rows.append({"assignment_id": aid, "distribution": dist, "trial": t, "k": k, "n_residual_flips": len(wf) + len(sf), "sqnr_db": db, "nonfinite_weights": nonfinite,
                         "restored_exactly": restored, "replicates_recorded_trial": same, "recorded_accuracy_drop_pp": drop})
    return rows


def summarize(rows: Sequence[dict], clean_db: float) -> Dict[str, dict]:
    """Per distribution: statistics of the SQNR after faults and of the loss relative to the clean assembled model."""
    out = {}
    for dist in sorted({r["distribution"] for r in rows}):
        rs = [r for r in rows if r["distribution"] == dist]
        v = np.array([r["sqnr_db"] for r in rs], dtype=float)
        fin = v[np.isfinite(v)]
        loss = clean_db - fin if np.isfinite(clean_db) else np.full_like(fin, np.nan)
        out[dist] = {"n_trials": len(rs), "n_nonfinite": int(np.sum(v == -np.inf)), "n_unchanged": int(np.sum(np.isfinite(v) & (np.abs(v - clean_db) < 1e-9))) if np.isfinite(clean_db) else 0,
                     "sqnr_median_db": float(np.median(fin)) if fin.size else float("nan"), "sqnr_mean_finite_db": float(fin.mean()) if fin.size else float("nan"),
                     "sqnr_p10_db": float(np.percentile(fin, 10)) if fin.size else float("nan"), "sqnr_worst_finite_db": float(fin.min()) if fin.size else float("nan"),
                     "loss_vs_clean_median_db": float(np.median(loss)) if loss.size else float("nan"), "loss_vs_clean_p90_db": float(np.percentile(loss, 90)) if loss.size else float("nan"),
                     "trials_losing_gt_1db": int(np.sum(loss > 1.0)) + int(np.sum(v == -np.inf)), "restored_all": all(r["restored_exactly"] for r in rs)}
    return out
