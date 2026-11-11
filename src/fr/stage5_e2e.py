"""Stage 5 end-to-end assembly: a complete model whose layers are stored in DIFFERENT encoded formats, and fault trials on it.

The optimiser (stage5.py) works with layer-additive estimates. Here a selected assignment (one datatype per layer, protection p = none) is
assembled into one model and measured as a whole: clean accuracy / loss, actual stored bytes (payload + scales), and random bit flips applied to the
STORED bytes of the assembled model (same injector rules as Stage 3: flips in encoded bytes, decode only affected elements, byte-exact restore, SHA-256
of the whole encoded state before / during / after).

Declared end-to-end fault distributions (weights are weight-only, biases/BatchNorm stay FP32 and are never faulted):
  weights_k8  k = 8 DISTINCT weights drawn uniformly from ALL quantizable weights of the assembled model; for each, one bit drawn uniformly from ALL bits of that
              weight's own stored encoding (16, 8 or 4 bits depending on the layer's datatype). This is the Stage 4 "global, any bit, k = 8" distribution applied to a
              mixed-format model.
  scales_k1   k = 1 scale word drawn uniformly from the float32 scale words of the INT-assigned layers, one of its 32 bits uniformly (extension, not in the paper);
              not defined if no layer is INT.
"""
from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from . import stage2 as S2
from . import stage3 as S3

DISTRIBUTIONS = ("weights_k8", "scales_k1")
SCHEMA_VERSION = "stage5-e2e-trial-v1"


class MixedStore:
    """Stored encoded state of an assembled model: layer l is stored in ``assignment[l]`` (copies of the frozen Stage 2 bytes)."""

    def __init__(self, assignment: Sequence[str], stores: Dict[str, S3.EncodedStore]):
        first = next(iter(stores.values()))
        if len(assignment) != len(first.names):
            raise ValueError("assignment length differs from the number of layers")
        self.names = list(first.names)
        self.shapes = list(first.shapes)
        self.n = np.array(first.n, copy=True)
        self.offsets = np.concatenate([[0], np.cumsum(self.n)])
        self.fmt_of = list(assignment)
        self.words, self.scales = [], []
        for l, f in enumerate(self.fmt_of):
            st = stores[f]
            if st.names != self.names:
                raise ValueError("stores disagree on layer names / order")
            self.words.append(np.array(st.words[l], copy=True))
            self.scales.append(None if st.scales[l] is None else np.array(st.scales[l], dtype=np.float32, copy=True).reshape(()))
        self.width = np.array([S3.SPECS[f]["width"] for f in self.fmt_of], dtype=np.int64)

    def hash(self) -> str:
        h = hashlib.sha256()
        for name, f, w, s in zip(self.names, self.fmt_of, self.words, self.scales):
            h.update(name.encode())
            h.update(f.encode())
            h.update(np.ascontiguousarray(w).data)
            if s is not None:
                h.update(s.tobytes())
        return h.hexdigest()

    def stored_bytes(self) -> dict:
        payload = sum(int(w.nbytes) for w in self.words)
        scale = sum(int(s.nbytes) for s in self.scales if s is not None)
        return {"payload_bytes": payload, "scale_bytes": scale, "total_bytes": payload + scale}

    def _view(self, l: int):
        return SimpleNamespace(fmt=self.fmt_of[l], words=self.words, scales=self.scales, shapes=self.shapes, n=self.n)

    def flip(self, layer: int, element: int, bit: int):
        f = self.fmt_of[layer]
        if not 0 <= element < int(self.n[layer]):
            raise IndexError(f"element {element} outside layer {self.names[layer]}")
        if not 0 <= bit < int(self.width[layer]):
            raise ValueError(f"bit {bit} outside the {int(self.width[layer])}-bit element of {f}")
        idx, pos = S3.bit_location(f, element, bit)
        w = self.words[layer]
        old = w[idx]
        w[idx] = old ^ w.dtype.type(1 << pos)
        return idx, int(old)

    def flip_scale(self, layer: int, bit: int) -> int:
        if self.scales[layer] is None:
            raise ValueError(f"{self.names[layer]} has no scale")
        u = self.scales[layer].reshape(1).view(np.uint32)
        old = int(u[0])
        u[0] = np.uint32(old ^ (1 << bit))
        return old


class MixedFaultInjector(S3.EncodedFaultInjector):
    """Same behaviour as the Stage 3 injector, with a per-layer datatype."""

    def __init__(self, model: nn.Module, store: MixedStore, codecs: Dict[str, S2.Codec], device):
        self.store, self.codecs, self.device = store, codecs, torch.device(device)
        self.codec = None
        self.model = copy.deepcopy(model).to(self.device).eval()
        params = dict(self.model.named_parameters())
        self.flat = []
        with torch.no_grad():
            for l, name in enumerate(store.names):
                w = S3.decode_layer(codecs[store.fmt_of[l]], store._view(l), l)
                params[name].copy_(torch.from_numpy(np.ascontiguousarray(w)).to(self.device))
                p = params[name].data
                if not p.is_contiguous():
                    raise ValueError(f"{name} is not contiguous")
                self.flat.append(p.view(-1))
        self._undo: List[tuple] = []

    def inject(self, weight_flips=(), scale_flips=()):
        if self._undo:
            raise RuntimeError("restore() must be called before the next inject()")
        by_layer: Dict[int, list] = {}
        for l, e, b in weight_flips:
            by_layer.setdefault(l, []).append((e, b))
        for l, items in by_layer.items():
            undo_bytes = [(l,) + self.store.flip(l, e, b) for e, b in items]
            elems = np.unique(np.array([e for e, _ in items], dtype=np.int64))
            new = S3.decode_elements(self.codecs[self.store.fmt_of[l]], self.store._view(l), l, elems)
            idx = torch.from_numpy(elems).to(self.device)
            old_vals = self.flat[l][idx].clone()
            self.flat[l][idx] = torch.from_numpy(new).to(self.device)
            self._undo.append(("w", l, undo_bytes, idx, old_vals))
        for l, b in scale_flips:
            old_scale = self.store.flip_scale(l, b)
            old_flat = self.flat[l].clone()
            with np.errstate(invalid="ignore", over="ignore"):
                full = S3.decode_layer(self.codecs[self.store.fmt_of[l]], self.store._view(l), l)
            self.flat[l].copy_(torch.from_numpy(np.ascontiguousarray(full).reshape(-1)).to(self.device))
            self._undo.append(("s", l, old_scale, old_flat))
        return {"weight_flips": len(weight_flips), "scale_flips": len(scale_flips)}


# ------------------------------------------------------------------------------------------------ sampling and trials
def draw_global_weight_flips(rng: np.random.Generator, store: MixedStore, k: int):
    """k DISTINCT weights uniformly over all quantizable weights; for each one bit uniformly over all bits of its own encoding."""
    total = int(store.n.sum())
    if k > total:
        raise ValueError("k larger than the number of weights")
    pick = rng.choice(total, size=k, replace=False)
    cum = np.cumsum(store.n)
    li = np.searchsorted(cum, pick, side="right")
    elem = pick - (cum[li] - store.n[li])
    bits = [int(rng.integers(0, int(store.width[l]))) for l in li]
    return [(int(l), int(e), b) for l, e, b in zip(li, elem, bits)]


def draw_scale_flips(rng: np.random.Generator, store: MixedStore, k: int):
    cand = [l for l, s in enumerate(store.scales) if s is not None]
    if not cand:
        raise ValueError("no INT layer: no scale words")
    if k > len(cand):
        raise ValueError("k larger than the number of scale words")
    pick = rng.choice(len(cand), size=k, replace=False)
    return [(int(cand[p]), int(rng.integers(0, 32))) for p in pick]


def has_scale_words(assignment: Sequence[str]) -> bool:
    return any(S3.SPECS[f]["kind"] == "int" for f in assignment)


def run_mixed_trial(inj: MixedFaultInjector, X, y, clean: dict, ctx: dict, *, assignment_id: str, dist: str, k: int, trial: int, base_seed: int,
                    explicit_weight_flips=None, explicit_scale_flips=None) -> dict:
    store = inj.store
    seed = S3.trial_seed(base_seed, f"mixed:{assignment_id}", "GLOBAL", dist, "all", k, trial)
    rng = np.random.default_rng(seed)
    if explicit_weight_flips is not None or explicit_scale_flips is not None:
        wf, sf = list(explicit_weight_flips or []), list(explicit_scale_flips or [])
    elif k == 0:
        wf, sf = [], []
    elif dist == "weights_k8":
        wf, sf = draw_global_weight_flips(rng, store, k), []
    elif dist == "scales_k1":
        wf, sf = [], draw_scale_flips(rng, store, k)
    else:
        raise ValueError(dist)
    h0 = store.hash()
    ev, h_fault, nonfinite_w = None, None, 0
    try:
        inj.inject(wf, sf)
        h_fault = store.hash()
        nonfinite_w = inj.count_nonfinite_weights()
        ev = S3.evaluate(inj.model, X, y)
    finally:
        inj.restore()
    h1 = store.hash()
    flags = []
    if nonfinite_w:
        flags.append("weights_nonfinite")
    if ev["activation_nonfinite"]:
        flags.append("activations_nonfinite")
    if ev["logit_rows_nonfinite"]:
        flags.append("logits_nonfinite")
    if ev["loss"] is None:
        flags.append("loss_nonfinite")
    flips = [[("w"), l, e, b, store.fmt_of[l]] for l, e, b in wf] + [["s", l, 0, b, store.fmt_of[l]] for l, b in sf]
    rec = {**ctx, "layer_id": -1, "layer_name": "GLOBAL", "format": f"mixed:{assignment_id}", "fault_group": "all", "scope_kind": "weights" if wf or not sf else "scales",
           "distribution": dist, "bit_index": [f[3] for f in flips], "fault_count": len(flips), "requested_k": k, "trial_index": trial, "seed": seed,
           "sample_count": int(y.numel()), "clean_accuracy": clean["accuracy"], "fault_accuracy": ev["accuracy"], "clean_loss": clean["loss"],
           "fault_loss": ev["loss"], "nonfinite_weight_count": nonfinite_w, "output_flag": ",".join(flags) if flags else "none", "failure": bool(flags),
           "encoded_hash_before": h0, "encoded_hash_faulted": h_fault, "encoded_hash_after_restore": h1, "restored_exactly": h0 == h1,
           "flip_applied_to_encoded_bytes": bool(flips) == (h_fault != h0), "flips": flips,   # [target, layer_id, element, bit, datatype of that layer]
           "assignment_id": assignment_id, "trial_id": f"{assignment_id}|{dist}|k{k}|t{trial}", "schema": SCHEMA_VERSION}
    return rec


# ------------------------------------------------------------------------------------------------ aggregation (pure functions)
def summarize_trials(recs: Sequence[dict]) -> dict:
    """Aggregate the trials of ONE (assignment, distribution): drop in pp, loss increase, failures by cause; exact counts."""
    n = len(recs)
    if n == 0:
        return {"n_trials": 0}
    drops = np.array([r["clean_accuracy"] - r["fault_accuracy"] for r in recs], dtype=np.float64)
    lossinc = np.array([r["fault_loss"] - r["clean_loss"] for r in recs if r["fault_loss"] is not None], dtype=np.float64)
    cause = lambda c: int(sum(c in r["output_flag"].split(",") for r in recs))
    out = {"n_trials": n, "sample_count": recs[0]["sample_count"], "requested_k": recs[0]["requested_k"], "clean_accuracy_subset": recs[0]["clean_accuracy"],
           "fault_accuracy_mean": float(np.mean([r["fault_accuracy"] for r in recs])),
           "drop_pp_mean": float(drops.mean()), "drop_pp_median": float(np.median(drops)), "drop_pp_std": float(drops.std(ddof=1)) if n > 1 else float("nan"),
           "drop_pp_p90": float(np.percentile(drops, 90)), "drop_pp_worst": float(drops.max()),
           "rel_drop_percent_mean": float(100 * drops.mean() / recs[0]["clean_accuracy"]) if recs[0]["clean_accuracy"] else float("nan"),
           "n_loss_finite": int(lossinc.size), "loss_increase_median": float(np.median(lossinc)) if lossinc.size else None,
           "failures": int(sum(r["failure"] for r in recs)), "failure_rate": float(np.mean([r["failure"] for r in recs])),
           "fail_weights_nonfinite": cause("weights_nonfinite"), "fail_activations_nonfinite": cause("activations_nonfinite"),
           "fail_logits_nonfinite": cause("logits_nonfinite"), "fail_loss_nonfinite": cause("loss_nonfinite")}
    return out


def pareto_front(points: Sequence[Sequence[float]]) -> List[int]:
    """Indices of non-dominated points (all coordinates minimised)."""
    P = np.asarray(points, dtype=np.float64)
    keep = []
    for i in range(len(P)):
        dom = np.all(P <= P[i], axis=1) & np.any(P < P[i], axis=1)
        if not dom.any():
            keep.append(i)
    return keep
