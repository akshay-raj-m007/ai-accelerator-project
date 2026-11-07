"""Random bit-flip fault injection into a MIXED-format weight buffer (E4M11 16-bit words and 8-bit INT8/FP8 words).

This extends the paper's protocol (Sec. 4, App. B.3; implemented for pure E4M11 in ``faults.py``) to the mixed
storage built by ``mixed.quantize_mixed``. The paper has no mixed-format experiment, so everything marked
"extension" below is a design choice of this project, not something the paper specifies.

Buffer. Every Conv2d/Linear weight is stored in the word format of its layer (``uint16`` for E4M11, ``uint8`` for the
8-bit formats). Layers are consecutive ranges of one global *weight* index space (the same space as ``faults.py``).
Scale metadata of INT8 / scaled FP8 is assumed fault-free (extension; stated in every result file).

Fault groups (paper Sec. 4.2). Bit positions per format:

=============  ===========  =======  ==============  ===============
format         exponent     sign     mantissa_msb    mantissa_lsb
=============  ===========  =======  ==============  ===============
e4m11          14-11        15       10-6            5-0       (App. B.3, as in ``faults.py``)
fp8 (E4M3)     6-3          7        2-1             0         (Sec. 4.2: upper ceil(M/2), lower floor(M/2))
int8           (empty)      7        6-3             2-0       (Sec. 4.2: value field = bits 6-0, upper ceil(7/2))
int4           (empty)      3        2-1             0         (4-bit words live in the low nibble of a uint8)
fp4 (E2M1)     2-1          3        0               (empty)
=============  ===========  =======  ==============  ===============

For INT8 the paper says the exponent group "is empty by construction". Bit 7 of an INT8 word is the two's-complement
sign position; flipping it changes the value by +-128 steps (it is *not* a sign-magnitude sign flip). The extra groups
``any`` (every bit of the word) and ``bitN`` (a single bit; only formats wide enough have it) are not in the paper.

Sampling (extension). One trial flips ``k`` bits, all applied before any decoding (flips on the same word XOR together).

``weights`` (the paper's wording, "k weights are selected uniformly at random"): k weights uniformly with replacement
    among the weights whose format has at least one bit in the group, then one of that weight's group bits uniformly.
    For the group ``any`` every weight is eligible, so the struck *weights* of a trial are identical for every
    storage layout (same seed): this gives paired comparisons between layouts.
``bits`` (physical DRAM view): k bits uniformly with replacement among all group bits of the whole buffer, so a
    weight is struck with probability proportional to its number of group bits (a 16-bit weight is twice as exposed
    as an 8-bit one).

For a pure-E4M11 buffer both reduce to the protocol of ``faults.py`` (the random streams differ, the distribution is
the same). A group such as ``exponent`` conditions on the bits that exist: in a mixed buffer INT8 layers cannot be hit.

Per trial we record accuracy and cross-entropy on the evaluation set as well as the prediction flips and the KL
divergence KL(softmax(clean) || softmax(faulty)) relative to the same layout's fault-free outputs (rare catastrophic
trials dominate means, so medians and percentiles are reported too).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import e4m11, formats8
from .faults import eval_tensors, trial_rng

SAMPLINGS = ("weights", "bits")
WORD_BITS = {"e4m11": 16, "int8_pt": 8, "int8_pc": 8, "fp8": 8, "fp8_scaled": 8,
             "int4_pt": 4, "int4_pc": 4, "fp4_scaled": 4}
PAPER_GROUP_NAMES = ("exponent", "sign", "mantissa_msb", "mantissa_lsb")

_E4M11 = {"exponent": (11, 12, 13, 14), "sign": (15,), "mantissa_msb": (6, 7, 8, 9, 10),
          "mantissa_lsb": (0, 1, 2, 3, 4, 5)}
_FP8 = {"exponent": (3, 4, 5, 6), "sign": (7,), "mantissa_msb": (1, 2), "mantissa_lsb": (0,)}
_INT8 = {"exponent": (), "sign": (7,), "mantissa_msb": (3, 4, 5, 6), "mantissa_lsb": (0, 1, 2)}
# 4-bit words (low nibble of a uint8; the high nibble is not part of the stored word and is never struck).
# Sec. 4.2 split applied to the value field: INT4 value bits 2-0 -> MSB = upper ceil(3/2) = 2 bits, LSB = 1 bit;
# FP4 E2M1: exponent 2-1, sign 3, mantissa 0 (M = 1 -> MSB = ceil(1/2) = 1 bit, LSB empty).
_INT4 = {"exponent": (), "sign": (3,), "mantissa_msb": (1, 2), "mantissa_lsb": (0,)}
_FP4 = {"exponent": (1, 2), "sign": (3,), "mantissa_msb": (0,), "mantissa_lsb": ()}
GROUP_BITS: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "e4m11": _E4M11, "fp8": _FP8, "fp8_scaled": _FP8, "int8_pt": _INT8, "int8_pc": _INT8,
    "int4_pt": _INT4, "int4_pc": _INT4, "fp4_scaled": _FP4}

CSV_FIELDS = ["group", "k", "trial", "accuracy", "cross_entropy", "correct", "total", "flips_vs_clean",
               "kl_vs_clean", "finite", "n_unique_weights", "flips_in_8bit", "max_abs_faulted_value"]


def group_bits(fmt: str, group: str) -> Tuple[int, ...]:
    """Bit positions of ``group`` in the word of format ``fmt`` (possibly empty)."""
    width = WORD_BITS[fmt]
    if group == "any":
        return tuple(range(width))
    m = re.fullmatch(r"bit(\d+)", group)
    if m:
        b = int(m.group(1))
        return (b,) if b < width else ()
    if group in GROUP_BITS[fmt]:
        return GROUP_BITS[fmt][group]
    raise ValueError(f"unknown fault group {group!r}; use {list(PAPER_GROUP_NAMES)}, 'any' or 'bitN'")


# ------------------------------------------------------------------------------------------ injector
class MixedFaultInjector:
    """Inject/restore bit flips on a model whose weights are stored as described by ``buffers``
    (the second return value of ``mixed.quantize_mixed``)."""

    def __init__(self, qmodel: nn.Module, buffers: "OrderedDict[str, dict]", e4m11_mode: str = "subnormal",
                 bias: int = e4m11.BIAS, all_ones: str = "clamp", device="cpu"):
        self.model = qmodel.to(device).eval()
        self.device = torch.device(device)
        self.mode, self.bias, self.all_ones = e4m11_mode, bias, all_ones
        self.names = list(buffers)
        self.fmts = [buffers[n]["format"] for n in self.names]
        for f in self.fmts:
            if f not in WORD_BITS:
                raise ValueError(f"unsupported format {f!r}")
        self.words = [np.array(buffers[n]["words"], copy=True) for n in self.names]
        self.scales = [buffers[n]["scale"] for n in self.names]
        self.shapes = [tuple(buffers[n]["shape"]) for n in self.names]
        self.starts = np.array([buffers[n]["global_start"] for n in self.names], dtype=np.int64)
        self.counts = np.array([buffers[n]["count"] for n in self.names], dtype=np.int64)
        self.width = np.array([WORD_BITS[f] for f in self.fmts], dtype=np.int64)
        params = dict(self.model.named_parameters())
        self.flat = []
        for n in self.names:
            p = params[n].data
            if not p.is_contiguous():
                raise ValueError(f"{n} is not contiguous")
            self.flat.append(p.view(-1))
        self.total = int(self.counts.sum())
        self.total_bits = int((self.counts * self.width).sum())
        self._saved: List[tuple] = []

    def assignment_hash(self) -> str:
        return hashlib.sha1("|".join(f"{n}:{f}" for n, f in zip(self.names, self.fmts)).encode()).hexdigest()[:12]

    # ---- sampling
    def _layer_mask(self, layer: Optional[str]) -> np.ndarray:
        if layer is None:
            return np.ones(len(self.names), dtype=bool)
        key = layer if layer in self.names else f"{layer}.weight"
        if key not in self.names:
            raise KeyError(f"layer {layer!r} not in buffer; have {self.names[:3]}...")
        m = np.zeros(len(self.names), dtype=bool)
        m[self.names.index(key)] = True
        return m

    def eligible(self, group: str, sampling: str, layer: Optional[str] = None):
        """Per-layer (bit tuples, number of group bits, eligible units: weights or bits)."""
        if sampling not in SAMPLINGS:
            raise ValueError(f"sampling must be one of {SAMPLINGS}")
        bits = [group_bits(f, group) for f in self.fmts]
        nb = np.array([len(b) for b in bits], dtype=np.int64)
        inc = self._layer_mask(layer)
        units = self.counts * ((nb > 0) & inc) * (nb if sampling == "bits" else 1)
        if units.sum() == 0:
            raise ValueError(f"group {group!r} has no bits in the selected layers of this buffer")
        return bits, nb, units

    def draw(self, rng: np.random.Generator, group: str, k: int, sampling: str = "weights",
             layer: Optional[str] = None):
        """k flips: (global weight indices int64, bit positions int64)."""
        bits, nb, units = self.eligible(group, sampling, layer)
        cum = np.cumsum(units)
        u = rng.integers(0, int(cum[-1]), size=k, dtype=np.int64)
        li = np.searchsorted(cum, u, side="right")
        off = u - (cum[li] - units[li])
        if sampling == "weights":
            local = off
            r = rng.random(k)
            pos = np.minimum((r * nb[li]).astype(np.int64), nb[li] - 1)
        else:
            local = off // nb[li]
            pos = off % nb[li]
        bit = np.array([bits[l][p] for l, p in zip(li, pos)], dtype=np.int64)
        return self.starts[li] + local, bit

    # ---- inject / restore
    def _decode(self, l: int, new_words: np.ndarray, local: np.ndarray) -> np.ndarray:
        fmt = self.fmts[l]
        if fmt == "e4m11":
            return e4m11.decode(new_words.astype(np.uint16), self.mode, self.all_ones, self.bias)
        w8 = new_words.astype(np.uint8)
        if fmt.startswith("int"):
            q = formats8.int_words_to_q(w8, WORD_BITS[fmt])
            scale = np.asarray(self.scales[l], dtype=np.float32)
            if scale.ndim == 0:
                return (q * scale).astype(np.float32)
            per_channel = int(np.prod(self.shapes[l][1:]))
            return (q * scale[local // per_channel]).astype(np.float32)
        if fmt == "fp4_scaled":
            return (formats8.fp4_decode(w8).astype(np.float64) * 2.0 ** int(self.scales[l])).astype(np.float32)
        vals = formats8.fp8_decode(w8, "clamp")
        if fmt == "fp8_scaled":
            return (vals.astype(np.float64) * 2.0 ** int(self.scales[l])).astype(np.float32)
        return vals

    def faulted(self, idx: np.ndarray, bit: np.ndarray):
        """(unique weight indices, layer index, local index, new words, new values) for a set of flips."""
        idx, bit = np.asarray(idx, dtype=np.int64), np.asarray(bit, dtype=np.int64)
        uniq, inv = np.unique(idx, return_inverse=True)
        li = np.searchsorted(self.starts, uniq, side="right") - 1
        if (bit >= self.width[np.searchsorted(self.starts, idx, side="right") - 1]).any():
            raise ValueError("bit position outside the word of the struck weight")
        mask = np.zeros(uniq.size, dtype=np.uint16)
        np.bitwise_xor.at(mask, inv, (np.uint16(1) << bit.astype(np.uint16)).astype(np.uint16))
        local = uniq - self.starts[li]
        vals = np.empty(uniq.size, dtype=np.float32)
        out_words = np.empty(uniq.size, dtype=np.uint16)
        for l in np.unique(li):
            sel = li == l
            nw = (self.words[l][local[sel]] ^ mask[sel].astype(self.words[l].dtype))
            out_words[sel] = nw
            vals[sel] = self._decode(int(l), nw, local[sel])
        return uniq, li, local, out_words, vals

    @torch.no_grad()
    def inject(self, idx: np.ndarray, bit: np.ndarray) -> dict:
        if self._saved:
            raise RuntimeError("restore() must be called before the next inject()")
        uniq, li, local, _, vals = self.faulted(idx, bit)
        for l in np.unique(li):
            sel = li == l
            flat = self.flat[l]
            loc = torch.from_numpy(local[sel]).to(self.device)
            self._saved.append((flat, loc, flat[loc].clone()))
            flat[loc] = torch.from_numpy(vals[sel]).to(self.device, flat.dtype)
        finite = vals[np.isfinite(vals)]
        hit_layers = np.searchsorted(self.starts, np.asarray(idx, dtype=np.int64), side="right") - 1
        return {"n_unique_weights": int(uniq.size),
                "flips_in_8bit": int((self.width[hit_layers] < 16).sum()),      # column name kept; = hits in low-bit layers
                "max_abs_faulted_value": float(np.abs(finite).max()) if finite.size else float("nan")}

    @torch.no_grad()
    def restore(self) -> None:
        for flat, loc, old in reversed(self._saved):
            flat[loc] = old
        self._saved = []

    def decode_all(self) -> "OrderedDict[str, np.ndarray]":
        """Fault-free decode of the stored words (what the model weights should equal)."""
        out = OrderedDict()
        for l, n in enumerate(self.names):
            out[n] = self._decode(l, self.words[l], np.arange(self.counts[l])).reshape(self.shapes[l])
        return out

    def clean_weights_unchanged(self) -> bool:
        ref = self.decode_all()
        return all(np.array_equal(ref[n].reshape(-1), self.flat[l].detach().cpu().numpy())
                   for l, n in enumerate(self.names))


# ------------------------------------------------------------------------------------------ evaluation
@torch.inference_mode()
def logits_of(model: nn.Module, X: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).float() for i in range(0, X.shape[0], batch_size)])


def trial_metrics(logits: torch.Tensor, clean_logits: torch.Tensor, y: torch.Tensor) -> dict:
    pred, cpred = logits.argmax(1), clean_logits.argmax(1)
    finite = bool(torch.isfinite(logits).all())
    ce = nn.functional.cross_entropy(logits, y, reduction="sum").item() / y.shape[0]
    if finite:
        lp_c = torch.log_softmax(clean_logits.double(), 1)
        lp_f = torch.log_softmax(logits.double(), 1)
        kl = (lp_c.exp() * (lp_c - lp_f)).sum(1).mean().item()
    else:
        kl = float("nan")
    return {"accuracy": 100.0 * (pred == y).float().mean().item(), "cross_entropy": ce,
            "correct": int((pred == y).sum()), "total": int(y.shape[0]),
            "flips_vs_clean": int((pred != cpred).sum()), "kl_vs_clean": kl, "finite": int(finite)}


# ------------------------------------------------------------------------------------------ campaign
def _read_done(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as fh:
        return {(r["group"], int(r["k"]), int(r["trial"])) for r in csv.DictReader(fh)}


def run_mixed_campaign(injector: MixedFaultInjector, X, y, out_dir: Path, groups: Iterable[str],
                       ks: Sequence[int], trials: int, seed: int, sampling: str = "weights",
                       layer: Optional[str] = None, meta: Optional[dict] = None, log=print) -> Path:
    """Run (resumably) every (group, k, trial); append one CSV row per finished trial."""
    if sampling not in SAMPLINGS:
        raise ValueError(f"sampling must be one of {SAMPLINGS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = out_dir / "trials.csv", out_dir / "campaign_meta.json"
    groups, skipped = list(groups), []
    for g in list(groups):
        if not any(group_bits(f, g) for f in WORD_BITS):          # typo: no format has such a group
            raise ValueError(f"unknown fault group {g!r}; use {list(PAPER_GROUP_NAMES)}, 'any' or 'bitN'")
        try:
            injector.eligible(g, sampling, layer)
        except ValueError:                                         # e.g. 'exponent' in an all-INT8 buffer
            groups.remove(g)
            skipped.append(g)
            log(f"[{g}/{sampling}] skipped: this layout has no {g} bits (paper: empty by construction)")
    meta = dict(meta or {}, seed=seed, layer=layer, sampling=sampling, trials=trials, ks=list(ks),
                groups=groups, skipped_groups=skipped, buffer_weights=injector.total, buffer_bits=injector.total_bits,
                assignment_hash=injector.assignment_hash(), mode=injector.mode, bias=injector.bias,
                all_ones=injector.all_ones, n_eval=int(X.shape[0]), scale_metadata_faults=False)
    if meta_path.exists():
        old = json.load(open(meta_path))
        for key in ("seed", "layer", "sampling", "mode", "bias", "all_ones", "n_eval", "assignment_hash"):
            if old.get(key) != meta.get(key):
                raise RuntimeError(f"cannot resume: {key} differs ({old.get(key)!r} vs {meta.get(key)!r}); "
                                   f"use a new output directory")
    clean_logits = logits_of(injector.model, X)
    meta["clean"] = eval_tensors(injector.model, X, y)
    json.dump(meta, open(meta_path, "w"), indent=2)
    done = _read_done(csv_path)
    new_file = not csv_path.exists()
    t0, n_run = time.time(), 0
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for g in groups:
            for k in ks:
                for t in range(trials):
                    if (g, k, t) in done:
                        continue
                    idx, b = injector.draw(trial_rng(seed, f"{g}|{sampling}", k, t), g, k, sampling, layer)
                    info = injector.inject(idx, b)
                    try:
                        m = trial_metrics(logits_of(injector.model, X), clean_logits, y)
                    finally:
                        injector.restore()
                    w.writerow({"group": g, "k": k, "trial": t, **m, **info})
                    fh.flush()
                    n_run += 1
                log(f"[{g}/{sampling} k={k}] done  ({n_run} trials this session, {time.time() - t0:.0f}s)")
    return csv_path


# ------------------------------------------------------------------------------------------ summaries
def load_trials(csv_path: Path) -> List[dict]:
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for key in ("k", "trial", "correct", "total", "flips_vs_clean", "finite", "n_unique_weights",
                    "flips_in_8bit"):
            r[key] = int(r[key])
        for key in ("accuracy", "cross_entropy", "kl_vs_clean", "max_abs_faulted_value"):
            r[key] = float(r[key])
    return rows


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, size=x.size, replace=True).mean() for _ in range(n_boot)])
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def summarize_trials(rows: List[dict], clean_acc: float, ref_clean_acc: float) -> List[dict]:
    cells: Dict[Tuple[str, int], List[dict]] = {}
    for r in rows:
        cells.setdefault((r["group"], r["k"]), []).append(r)
    out = []
    for (g, k), rs in sorted(cells.items()):
        acc = np.array([r["accuracy"] for r in rs])
        kl = np.array([r["kl_vs_clean"] for r in rs])
        lo, hi = bootstrap_ci(acc)
        out.append({
            "group": g, "k": k, "n_trials": len(rs), "mean_acc": float(acc.mean()), "std_acc": float(acc.std()),
            "median_acc": float(np.median(acc)), "p10_acc": float(np.quantile(acc, 0.10)),
            "min_acc": float(acc.min()), "ci95_mean_acc": [lo, hi],
            "drop_vs_own_clean_pp": float(clean_acc - acc.mean()),
            "drop_vs_ref_clean_pp": float(ref_clean_acc - acc.mean()),
            "mean_kl": float(np.nanmean(kl)) if np.isfinite(kl).any() else float("nan"),
            "median_kl": float(np.nanmedian(kl)) if np.isfinite(kl).any() else float("nan"),
            "n_nonfinite": int(sum(1 for r in rs if not r["finite"])),
            "mean_flips_in_8bit": float(np.mean([r["flips_in_8bit"] for r in rs]))})
    return out


def collect(campaign_dirs: Sequence[Path]) -> List[dict]:
    """Load meta + trials of every campaign directory (those without results are skipped)."""
    out = []
    for d in campaign_dirs:
        d = Path(d)
        if not (d / "trials.csv").exists() or not (d / "campaign_meta.json").exists():
            continue
        meta = json.load(open(d / "campaign_meta.json"))
        out.append({"dir": d, "slug": d.name, "meta": meta, "rows": load_trials(d / "trials.csv")})
    return out


def render_comparison(camps: List[dict], reference: str = "e4m11", ref_clean_acc: Optional[float] = None) -> str:
    """Markdown comparison across storage layouts, one table per (group, sampling, k)."""
    if not camps:
        return "no campaigns found\n"
    ref = next((c for c in camps if c["slug"] == reference), None)
    ref_clean = ref_clean_acc if ref_clean_acc is not None else \
        (ref["meta"]["clean"]["accuracy_percent"] if ref else camps[0]["meta"]["clean"]["accuracy_percent"])
    L = ["# Fault robustness of mixed E4M11 / 8-bit storage", "",
         f"* reference layout: `{reference}` (clean accuracy {ref_clean:.2f}%); 'drop vs E4M11' always uses that value",
         "* scale metadata of INT8 / scaled-FP8 layers is assumed fault-free; bit groups per format and the two "
         "sampling models are defined in `fr/faults_mixed.py` (project extension, not the paper)",
         "* mean accuracy of faulty runs is heavy-tailed (rare catastrophic trials): read the median, p10 and "
         "bootstrap interval next to the mean", ""]
    keys = sorted({(r["group"], c["meta"]["sampling"], r["k"]) for c in camps for r in c["rows"]})
    sums = {c["slug"]: summarize_trials(c["rows"], c["meta"]["clean"]["accuracy_percent"], ref_clean) for c in camps}
    for g, s, k in keys:
        L += [f"## group `{g}`, sampling `{s}`, k = {k}", "",
              "| layout | bits/weight | clean acc | mean acc | 95% CI of mean | median | p10 | drop vs E4M11 (pp) | "
              "median KL | mean KL | non-finite | 8-bit hits/trial |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        rows_for = []
        for c in camps:
            if c["meta"]["sampling"] != s:
                continue
            rec = next((r for r in sums[c["slug"]] if r["group"] == g and r["k"] == k), None)
            if rec is None:
                continue
            bits = c["meta"].get("avg_bits_payload", c["meta"]["buffer_bits"] / c["meta"]["buffer_weights"])
            rows_for.append((-bits, c, rec, bits))
        for _, c, r, bits in sorted(rows_for, key=lambda t: (t[0], t[1]["slug"])):
            L.append(f"| {c['slug']} | {bits:.2f} | {c['meta']['clean']['accuracy_percent']:.2f} | "
                     f"{r['mean_acc']:.2f} | [{r['ci95_mean_acc'][0]:.2f}, {r['ci95_mean_acc'][1]:.2f}] | "
                     f"{r['median_acc']:.2f} | {r['p10_acc']:.2f} | {r['drop_vs_ref_clean_pp']:.2f} | "
                     f"{r['median_kl']:.2e} | {r['mean_kl']:.2e} | {r['n_nonfinite']} | "
                     f"{r['mean_flips_in_8bit']:.2f} |")
        absent = [c["slug"] for c in camps if c["meta"]["sampling"] == s and c["meta"].get("skipped_groups")
                  and g in c["meta"]["skipped_groups"]]
        if absent:
            L += ["", f"Not applicable (the layout has no `{g}` bits): {', '.join(absent)}"]
        # paired differences against the reference layout (same struck weights: group 'any', sampling 'weights')
        if ref is not None and g == "any" and s == "weights" and ref["meta"]["sampling"] == s:
            ra = {r["trial"]: r["accuracy"] for r in ref["rows"] if r["group"] == g and r["k"] == k}
            L += ["", f"Paired difference in accuracy vs `{reference}` (same struck weights per trial; "
                       "negative = worse than the reference):", "",
                  "| layout | trials | mean diff (pp) | s.e. | trials worse / equal / better |", "|---|---|---|---|---|"]
            for _, c, _, _ in sorted(rows_for, key=lambda t: (t[0], t[1]["slug"])):
                if c["slug"] == reference:
                    continue
                d = np.array([r["accuracy"] - ra[r["trial"]] for r in c["rows"]
                              if r["group"] == g and r["k"] == k and r["trial"] in ra])
                if d.size:
                    L.append(f"| {c['slug']} | {d.size} | {d.mean():+.3f} | {d.std(ddof=1) / np.sqrt(d.size):.3f} | "
                             f"{int((d < -1e-9).sum())} / {int((abs(d) <= 1e-9).sum())} / {int((d > 1e-9).sum())} |")
        L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------ specs
def slugify(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", spec).strip("_")


def normalize_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Layout spec -> (slug, config_id). Accepted forms:

    ``e4m11``                    every layer E4M11 (the baseline)
    ``all:<arm>``                every layer in the 8-bit format ``arm``
    ``<arm>:T12:product``        saved selections: ``<arm>:T<bits>:<method>[:<param>]`` with method in
                                 lambda (param = lambda), product, worst, random (param = draw)
    ``<arm>|T12|product|``       the exact config id of ``selections.json``
    """
    if spec == "e4m11":
        return "e4m11", None
    if spec.startswith("all:"):
        arm = spec.split(":", 1)[1]
        if arm not in formats8.ALL_LOW:
            raise ValueError(f"unknown low-bit format {arm!r}")
        return f"all_{arm}", None
    parts = spec.split("|") if "|" in spec else spec.split(":")
    if len(parts) < 3:
        raise ValueError(f"cannot parse layout spec {spec!r}")
    arm, t, method = parts[0], parts[1].lstrip("T"), parts[2]
    param = parts[3] if len(parts) > 3 else ""
    if arm not in formats8.ALL_LOW or method not in ("lambda", "product", "worst", "random"):
        raise ValueError(f"cannot parse layout spec {spec!r}")
    tv = float(t)
    cid = f"{arm}|T{tv:g}|{method}|{param}"
    return slugify(f"{arm}_T{tv:g}_{method}" + (f"_{param}" if param else "")), cid


def resolve_assignment(spec: str, selections: dict, layer_names: Sequence[str]) -> Tuple[str, Dict[str, str], str]:
    """(slug, {layer: format}, label) for a layout spec."""
    slug, cid = normalize_spec(spec)
    if spec == "e4m11":
        return slug, {}, "all E4M11"
    if spec.startswith("all:"):
        arm = spec.split(":", 1)[1]
        return slug, {n: arm for n in layer_names}, f"all {arm}"
    if cid not in selections:
        raise KeyError(f"{cid!r} not found in selections.json (run the sweep with that target/method first)")
    arm = cid.split("|")[0]
    return slug, {n: arm for n in selections[cid]["layers8"]}, cid
