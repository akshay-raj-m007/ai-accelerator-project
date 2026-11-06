"""Random bit-flip fault injection into the packed E4M11 weight buffer (paper Sec. 4, App. B.3).

Protocol (one *experiment* = one fault group, one flip count k, 100 independent trials):

* All quantized weights live in ONE flat ``uint16`` buffer ("the full weight buffer", App. B.3).
  Layers are just index ranges of that buffer (same global index space as the Day-9 notebook).
* A trial draws ``k`` weight indices uniformly at random from the buffer, **with replacement**
  (App. B.3), and for each one a bit uniformly at random from the target group
  (Sec. 4.3: "one bit is chosen uniformly at random from the target group"), then flips it with
  XOR. All flips are applied before any decoding; flips that hit the same weight combine by XOR.
* The corrupted words are decoded to FP32 and the model is evaluated on the test set.
  The clean buffer/weights are restored after each trial, so faults never accumulate.

Groups (App. B.3): exponent bits 14-11, sign bit 15, mantissa MSBs 10-6, mantissa LSBs 5-0.
Extra, clearly non-paper groups: ``any`` (all 16 bits) and ``bitN`` (a single bit, e.g. ``bit14``
reproduces the Day-9 notebook). ``--layer`` restricts the index range to one layer (Day-9 style);
the paper protocol uses the whole buffer.

Efficiency: only the touched words are decoded and written into the model parameters, then
restored, so a trial costs one forward pass over the test set.
"""
from __future__ import annotations

import csv
import json
import re
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import e4m11

PAPER_GROUPS: Dict[str, Tuple[int, ...]] = OrderedDict([
    ("exponent", (11, 12, 13, 14)),
    ("sign", (15,)),
    ("mantissa_msb", (6, 7, 8, 9, 10)),     # App. B.3 (Sec. 4.2 says upper ceil(M/2)=6 bits)
    ("mantissa_lsb", (0, 1, 2, 3, 4, 5)),
])
EXTRA_GROUPS: Dict[str, Tuple[int, ...]] = {"any": tuple(range(16))}
CSV_FIELDS = ["group", "k", "trial", "accuracy", "cross_entropy", "correct", "total",
              "n_unique_weights", "max_abs_faulted_value"]


def parse_group(name: str) -> Tuple[str, Tuple[int, ...]]:
    if name in PAPER_GROUPS:
        return name, PAPER_GROUPS[name]
    if name in EXTRA_GROUPS:
        return name, EXTRA_GROUPS[name]
    m = re.fullmatch(r"bit(\d+)", name)
    if m and int(m.group(1)) < 16:
        return name, (int(m.group(1)),)
    raise ValueError(f"unknown fault group {name!r}; use {list(PAPER_GROUPS)}, 'any' or 'bitN'")


def trial_rng(seed: int, group: str, k: int, trial: int) -> np.random.Generator:
    """Independent, reproducible stream per (seed, group, k, trial)."""
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(group.encode()), k, trial]))


def draw_flips(rng: np.random.Generator, lo: int, hi: int, k: int, bits: Sequence[int]):
    """k indices in [lo, hi) with replacement, k bits uniformly from ``bits``."""
    idx = rng.integers(lo, hi, size=k, dtype=np.int64)
    b = rng.choice(np.asarray(bits, dtype=np.int64), size=k)
    return idx, b


def combine_flips(idx: np.ndarray, bits: np.ndarray):
    """Collapse duplicate indices: returns (unique_idx, xor_mask uint16) (flips on one word XOR together)."""
    uniq, inv = np.unique(idx, return_inverse=True)
    mask = np.zeros(uniq.size, dtype=np.uint16)
    np.bitwise_xor.at(mask, inv, (np.uint16(1) << bits.astype(np.uint16)).astype(np.uint16))
    return uniq, mask


class FaultInjector:
    """Inject/restore bit flips on a quantized model whose weights are described by ``buffers``."""

    def __init__(self, qmodel: nn.Module, buffers: "OrderedDict[str, dict]", mode: str = "subnormal",
                 bias: int = e4m11.BIAS, all_ones: str = "clamp", device="cpu"):
        self.model = qmodel.to(device).eval()
        self.device = torch.device(device)
        self.mode, self.bias, self.all_ones = mode, bias, all_ones
        self.names = list(buffers)
        self.words = np.concatenate([buffers[n]["words"] for n in self.names]).astype(np.uint16)
        self.starts = np.array([buffers[n]["global_start"] for n in self.names], dtype=np.int64)
        self.counts = np.array([buffers[n]["count"] for n in self.names], dtype=np.int64)
        params = dict(self.model.named_parameters())
        self.flat = []
        for n in self.names:
            p = params[n].data
            if not p.is_contiguous():
                raise ValueError(f"{n} is not contiguous")
            self.flat.append(p.view(-1))
        self.total = int(self.words.size)
        self._saved: List[tuple] = []

    # ---- index space
    def layer_range(self, layer: Optional[str]) -> Tuple[int, int]:
        if layer is None:
            return 0, self.total
        key = layer if layer in self.names else f"{layer}.weight"
        if key not in self.names:
            raise KeyError(f"layer {layer!r} not in buffer; have {self.names[:3]}...")
        i = self.names.index(key)
        return int(self.starts[i]), int(self.starts[i] + self.counts[i])

    def locate(self, idx: np.ndarray):
        li = np.searchsorted(self.starts, idx, side="right") - 1
        return li, idx - self.starts[li]

    # ---- inject / restore
    def faulted_values(self, idx: np.ndarray, bits: np.ndarray):
        uniq, mask = combine_flips(idx, bits)
        new_words = (self.words[uniq] ^ mask).astype(np.uint16)
        new_vals = e4m11.decode(new_words, self.mode, self.all_ones, self.bias)
        return uniq, new_words, new_vals

    @torch.no_grad()
    def inject(self, idx: np.ndarray, bits: np.ndarray) -> dict:
        if self._saved:
            raise RuntimeError("restore() must be called before the next inject()")
        uniq, _, new_vals = self.faulted_values(idx, bits)
        li, local = self.locate(uniq)
        for l in np.unique(li):
            sel = li == l
            flat = self.flat[l]
            loc = torch.from_numpy(local[sel]).to(self.device)
            self._saved.append((flat, loc, flat[loc].clone()))
            flat[loc] = torch.from_numpy(new_vals[sel]).to(self.device, flat.dtype)
        finite = new_vals[np.isfinite(new_vals)]
        return {"n_unique_weights": int(uniq.size),
                "max_abs_faulted_value": float(np.abs(finite).max()) if finite.size else float("nan")}

    @torch.no_grad()
    def restore(self) -> None:
        for flat, loc, old in reversed(self._saved):
            flat[loc] = old
        self._saved = []

    def clean_weights_unchanged(self) -> bool:
        """Debug check: model weights equal the decoded clean buffer."""
        full = e4m11.decode(self.words, self.mode, self.all_ones, self.bias)
        got = torch.cat([f.detach().cpu() for f in self.flat]).numpy()
        return bool(np.array_equal(full, got))


# --------------------------------------------------------------------------- evaluation
def load_tensors(loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


@torch.inference_mode()
def eval_tensors(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 1000) -> dict:
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, X.shape[0], batch_size):
        logits = model(X[i:i + batch_size])
        loss += nn.functional.cross_entropy(logits.float(), y[i:i + batch_size], reduction="sum").item()
        correct += (logits.argmax(1) == y[i:i + batch_size]).sum().item()
    n = X.shape[0]
    return {"accuracy_percent": 100.0 * correct / n, "correct": correct, "total": n,
            "cross_entropy": loss / n}


# --------------------------------------------------------------------------- campaign
def _read_done(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as fh:
        return {(r["group"], int(r["k"]), int(r["trial"])) for r in csv.DictReader(fh)}


def run_campaign(injector: FaultInjector, X, y, out_dir: Path, groups: Iterable[str],
                 ks: Sequence[int], trials: int, seed: int, layer: Optional[str] = None,
                 meta: Optional[dict] = None, log=print) -> Path:
    """Run (resumably) every (group, k, trial); append one CSV row per finished trial."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = out_dir / "trials.csv", out_dir / "campaign_meta.json"
    groups = [parse_group(g) for g in groups]
    lo, hi = injector.layer_range(layer)
    meta = dict(meta or {}, seed=seed, layer=layer, trials=trials, ks=list(ks),
                groups=[g for g, _ in groups], buffer_words=injector.total, index_range=[lo, hi],
                mode=injector.mode, bias=injector.bias, all_ones=injector.all_ones, n_eval=int(X.shape[0]))
    if meta_path.exists():
        old = json.load(open(meta_path))
        for key in ("seed", "layer", "mode", "bias", "all_ones", "n_eval", "index_range"):
            if old.get(key) != meta.get(key):
                raise RuntimeError(f"cannot resume: {key} differs ({old.get(key)!r} vs {meta.get(key)!r}); "
                                   f"use a new --run-name")
    meta["clean"] = eval_tensors(injector.model, X, y)
    json.dump(meta, open(meta_path, "w"), indent=2)
    done = _read_done(csv_path)
    new_file = not csv_path.exists()
    t0, n_run = time.time(), 0
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for gname, bits in groups:
            for k in ks:
                for t in range(trials):
                    if (gname, k, t) in done:
                        continue
                    idx, b = draw_flips(trial_rng(seed, gname, k, t), lo, hi, k, bits)
                    info = injector.inject(idx, b)
                    try:
                        m = eval_tensors(injector.model, X, y)
                    finally:
                        injector.restore()
                    w.writerow({"group": gname, "k": k, "trial": t, "accuracy": m["accuracy_percent"],
                                "cross_entropy": m["cross_entropy"], "correct": m["correct"],
                                "total": m["total"], **info})
                    fh.flush()
                    n_run += 1
                log(f"[{gname} k={k}] done  ({n_run} trials this session, {time.time() - t0:.0f}s)")
    return csv_path


def summarize(csv_path: Path, clean_acc: float, fp32_acc: Optional[float] = None) -> List[dict]:
    rows: Dict[Tuple[str, int], List[float]] = {}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault((r["group"], int(r["k"])), []).append(float(r["accuracy"]))
    out = []
    for (g, k), accs in rows.items():
        a = np.asarray(accs)
        rec = {"group": g, "k": k, "n_trials": int(a.size), "mean_acc": float(a.mean()),
               "std_acc": float(a.std(ddof=0)), "min_acc": float(a.min()), "max_acc": float(a.max()),
               "mean_drop_vs_clean_e4m11_pp": float(clean_acc - a.mean())}
        if fp32_acc is not None:
            rec["mean_drop_vs_fp32_pp"] = float(fp32_acc - a.mean())
        out.append(rec)
    return out


def render_summary_md(summary: List[dict], meta: dict, fp32_acc: Optional[float], targets: Optional[dict]) -> str:
    L = ["# Fault-injection summary", "",
         f"* format: E4M11 mode `{meta['mode']}`, bias {meta['bias']}, all-ones decode `{meta['all_ones']}`",
         f"* clean E4M11 accuracy: {meta['clean']['accuracy_percent']:.2f}% on {meta['n_eval']} images"
         + (f"; FP32: {fp32_acc:.2f}%" if fp32_acc is not None else ""),
         f"* trials per cell: {meta['trials']}; seed {meta['seed']}; fault index range {meta['index_range']}"
         + (f" (layer `{meta['layer']}`)" if meta.get("layer") else " (whole buffer)"), "",
         "| group | k | trials | mean acc % | std | min | max | drop vs clean E4M11 (pp) |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(summary, key=lambda r: (r["group"], r["k"])):
        L.append(f"| {r['group']} | {r['k']} | {r['n_trials']} | {r['mean_acc']:.2f} | {r['std_acc']:.2f} | "
                 f"{r['min_acc']:.2f} | {r['max_acc']:.2f} | {r['mean_drop_vs_clean_e4m11_pp']:.2f} |")
    if targets:
        L += ["", "Paper reference at k=8 (ResNet-18, CIFAR-10; baseline of 'drop' not stated in the paper):", "",
              "| group | paper | ours (drop vs clean E4M11) | source |", "|---|---|---|---|"]
        by = {(r["group"], r["k"]): r for r in summary}
        for g, e in targets.items():
            r = by.get((g, 8))
            ours = "n/a" if r is None else f"{r['mean_drop_vs_clean_e4m11_pp']:.2f} pp"
            paper = f"{e['value']} %" if e.get("value") is not None else f"< {e.get('upper_bound')} %"
            L.append(f"| {g} | {paper} | {ours} | {e.get('source', '')} |")
    return "\n".join(L) + "\n"
