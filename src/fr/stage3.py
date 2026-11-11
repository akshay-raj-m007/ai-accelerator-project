"""J-FAMP runbook Stage 3: bit-fault injection into the ACTUAL ENCODED weights (Sections 5.1 and 5.3).

The injector never touches FP32 values. It owns a mutable copy of the stored words (uint16 / uint8 / packed INT4 bytes and the
float32 INT scales, i.e. what Stage 2 saved), XOR-flips bits of those stored bytes, decodes only the affected elements from the
stored bytes, writes them into the evaluation model, and restores the stored bytes afterwards. Hashes of the whole encoded state
before / during / after a trial prove that the flip happened in the encoded representation and that restoration is exact.

Bit maps (positions counted from the least-significant bit of one element's stored word):

  e4m11_sn / e4m11_ftz   16 bit  sign 15 | exponent 11-14 | fraction 0-10                (S EEEE MMMMMMMMMMM)
  fp16                   16 bit  sign 15 | exponent 10-14 | fraction 0-9                 (S EEEEE MMMMMMMMMM)
  fp8_e4m3fn              8 bit  sign  7 | exponent  3-6  | fraction 0-2                 (S EEEE MMM)
  int8_pt                 8 bit  sign  7 (two's complement MSB, weight -128) | value 0-6 (weights +1..+64)
  int4_pt                 4 bit  sign  3 (two's complement MSB, weight -8)   | value 0-2 (weights +1..+4); packed two per byte
  INT scale (float32)    32 bit  sign 31 | exponent 23-30 | fraction 0-22       (optional extension scope `scales`)

INT4 packing: element 2j is the LOW nibble (bits 0-3) of byte j, element 2j+1 the HIGH nibble (bits 4-7); an odd element count
leaves one zero padding nibble that is never a fault target. Physical location of bit b of element e: byte e // 2, bit
(e % 2) * 4 + b.

Sampling policy (runbook 5.1): the k struck elements of a trial are DISTINCT (drawn without replacement from the scope), the bit
of each is drawn uniformly from the target group independently, trials are independent (one derived seed per trial, recorded).

Non-finite policy: an image whose logits contain NaN/Inf counts as INCORRECT; the loss is recorded as null when any logit is
non-finite; the trial is never dropped and carries an ``output_flag``.
"""
from __future__ import annotations

import hashlib
import math
import struct
import zlib
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import e4m11, formats8
from . import stage2 as S2

INJECTOR_VERSION = "stage3-injector-v1"

# --------------------------------------------------------------------------------------------- bit maps
SPECS: "OrderedDict[str, dict]" = OrderedDict([
    ("e4m11_sn", {"kind": "float", "width": 16, "e_bits": 4, "m_bits": 11, "bias": 7, "word_dtype": np.uint16,
                  "fields": {"sign": (15,), "exponent": (11, 12, 13, 14), "fraction": tuple(range(0, 11))}}),
    ("fp16", {"kind": "float", "width": 16, "e_bits": 5, "m_bits": 10, "bias": 15, "word_dtype": np.uint16,
              "fields": {"sign": (15,), "exponent": (10, 11, 12, 13, 14), "fraction": tuple(range(0, 10))}}),
    ("fp8_e4m3fn", {"kind": "float", "width": 8, "e_bits": 4, "m_bits": 3, "bias": 7, "word_dtype": np.uint8,
                    "fields": {"sign": (7,), "exponent": (3, 4, 5, 6), "fraction": (0, 1, 2)}}),
    ("int8_pt", {"kind": "int", "width": 8, "word_dtype": np.uint8,
                 "fields": {"sign": (7,), "value": tuple(range(0, 7))}}),
    ("int4_pt", {"kind": "int", "width": 4, "word_dtype": np.uint8,
                 "fields": {"sign": (3,), "value": (0, 1, 2)}}),
    ("e4m11_ftz", {"kind": "float", "width": 16, "e_bits": 4, "m_bits": 11, "bias": 7, "word_dtype": np.uint16,
                   "fields": {"sign": (15,), "exponent": (11, 12, 13, 14), "fraction": tuple(range(0, 11))}}),
])
SCALE_FIELDS = {"sign": (31,), "exponent": tuple(range(23, 31)), "fraction": tuple(range(0, 23))}
SCALE_FORMATS = ("int8_pt", "int4_pt")


def group_bits(fmt: str, group: str, scope_kind: str = "weights") -> Tuple[int, ...]:
    """Bit positions of a fault group. ``all`` = every bit of the element. Empty tuple if the format lacks the group."""
    fields = SCALE_FIELDS if scope_kind == "scales" else SPECS[fmt]["fields"]
    if group == "all":
        return tuple(sorted(b for bits in fields.values() for b in bits))
    return tuple(fields.get(group, ()))


def format_groups(fmt: str, scope_kind: str = "weights") -> List[str]:
    fields = SCALE_FIELDS if scope_kind == "scales" else SPECS[fmt]["fields"]
    return list(fields) + ["all"]


def field_of_bit(fmt: str, bit: int, scope_kind: str = "weights") -> str:
    fields = SCALE_FIELDS if scope_kind == "scales" else SPECS[fmt]["fields"]
    for name, bits in fields.items():
        if bit in bits:
            return name
    raise ValueError(f"bit {bit} not in {fmt}")


def bit_location(fmt: str, element: int, bit: int) -> Tuple[int, int]:
    """(index into the stored array, bit position inside that stored array element)."""
    if fmt == "int4_pt":
        return element // 2, (element % 2) * 4 + bit          # independent of stage2.int4_bit_location (cross-checked in tests)
    return element, bit


# --------------------------------------------------------------------------------------------- reference decoders
def ref_decode_float(fmt: str, word: int) -> float:
    """First-principles value of one stored float word (pure Python, independent of the NumPy codecs)."""
    sp = SPECS[fmt]
    E, M, bias = sp["e_bits"], sp["m_bits"], sp["bias"]
    s = (word >> (E + M)) & 1
    es = (word >> M) & ((1 << E) - 1)
    m = word & ((1 << M) - 1)
    if fmt in ("e4m11_sn", "e4m11_ftz"):
        if es == 0:
            mag = m * 2.0 ** (1 - bias - M) if fmt == "e4m11_sn" else 0.0
        else:
            mag = (1.0 + m / 2.0 ** M) * 2.0 ** (min(es, 14) - bias)         # reserved es=15 is read as 14
    elif fmt == "fp16":
        if es == 31:
            mag = math.inf if m == 0 else math.nan
        elif es == 0:
            mag = m * 2.0 ** -24
        else:
            mag = (1.0 + m / 1024.0) * 2.0 ** (es - 15)
    elif fmt == "fp8_e4m3fn":
        if es == 15 and m == 7:
            mag = math.nan
        elif es == 0:
            mag = m * 2.0 ** -9
        else:
            mag = (1.0 + m / 8.0) * 2.0 ** (es - 7)
    else:
        raise ValueError(fmt)
    return -mag if s else mag


def ref_decode_int(bits: int, code: int, scale: float) -> float:
    """Two's complement code (0 .. 2^bits-1) times the float32 scale, rounded to float32 like the codec."""
    q = code - (1 << bits) if code >= (1 << (bits - 1)) else code
    return float(np.float32(q) * np.float32(scale))


def float32_from_bits(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def same_value(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


# --------------------------------------------------------------------------------------------- encoded store
class EncodedStore:
    """Mutable copy of the stored encoded weights of one format (what is physically in memory)."""

    def __init__(self, fmt: str, encoded: Mapping[str, dict]):
        self.fmt = fmt
        self.names = list(encoded)
        self.words = [np.array(e["words"], copy=True) for e in encoded.values()]
        self.scales = [None if e["scale"] is None else np.array(e["scale"], dtype=np.float32, copy=True).reshape(())
                       for e in encoded.values()]
        self.shapes = [tuple(e["shape"]) for e in encoded.values()]
        self.n = np.array([int(e["n"]) for e in encoded.values()], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.n)])

    # ---- hashing of the whole stored state (words + scales)
    def hash(self) -> str:
        h = hashlib.sha256()
        for name, w, s in zip(self.names, self.words, self.scales):
            h.update(name.encode())
            h.update(np.ascontiguousarray(w).data)
            if s is not None:
                h.update(s.tobytes())
        return h.hexdigest()

    # ---- element access
    def code(self, layer: int, element: int) -> int:
        """Stored code of one element (INT4: the nibble)."""
        if self.fmt == "int4_pt":
            return (int(self.words[layer][element // 2]) >> ((element % 2) * 4)) & 0xF
        return int(self.words[layer][element])

    def flip(self, layer: int, element: int, bit: int) -> Tuple[int, int]:
        """XOR one bit of the STORED bytes. Returns (stored index, previous stored byte/word) for the undo."""
        width = SPECS[self.fmt]["width"]
        if not 0 <= element < int(self.n[layer]):
            raise IndexError(f"element {element} outside layer {self.names[layer]} (n={int(self.n[layer])})")
        if not 0 <= bit < width:
            raise ValueError(f"bit {bit} outside the {width}-bit element")
        idx, pos = bit_location(self.fmt, element, bit)
        w = self.words[layer]
        old = w[idx]
        w[idx] = old ^ w.dtype.type(1 << pos)
        return idx, int(old)

    def flip_scale(self, layer: int, bit: int) -> int:
        """XOR one bit of the stored float32 scale; returns the previous 32-bit pattern for an exact undo."""
        if self.scales[layer] is None:
            raise ValueError(f"{self.names[layer]} has no scale")
        if not 0 <= bit < 32:
            raise ValueError("scale bit must be 0..31")
        u = self.scales[layer].reshape(1).view(np.uint32)
        old = int(u[0])
        u[0] = np.uint32(old ^ (1 << bit))
        return old

    def to_encoded(self) -> "OrderedDict[str, dict]":
        return OrderedDict((n, {"words": self.words[i], "scale": self.scales[i], "shape": self.shapes[i],
                                "n": int(self.n[i]), "n_clipped": 0}) for i, n in enumerate(self.names))


def decode_elements(codec: S2.Codec, store: EncodedStore, layer: int, elements: np.ndarray) -> np.ndarray:
    """Decode only ``elements`` of one layer from the STORED bytes."""
    el = np.asarray(elements, dtype=np.int64)
    fmt = store.fmt
    if fmt == "int4_pt":
        codes = ((store.words[layer][el // 2] >> ((el % 2) * 4).astype(np.uint8)) & np.uint8(0xF)).astype(np.uint8)
        return (formats8.int_words_to_q(codes, 4) * np.float32(store.scales[layer])).astype(np.float32)
    if fmt == "int8_pt":
        return (formats8.int_words_to_q(store.words[layer][el], 8) * np.float32(store.scales[layer])).astype(np.float32)
    return codec.decode({"words": store.words[layer][el], "shape": (el.size,)})


def decode_layer(codec: S2.Codec, store: EncodedStore, layer: int) -> np.ndarray:
    enc = {"words": store.words[layer], "scale": store.scales[layer], "shape": store.shapes[layer], "n": int(store.n[layer])}
    return codec.decode(enc)


# --------------------------------------------------------------------------------------------- sampling
def trial_seed(base_seed: int, fmt: str, scope_name: str, scope_kind: str, group: str, k: int, trial: int) -> int:
    ss = np.random.SeedSequence([int(base_seed), zlib.crc32(fmt.encode()), zlib.crc32(scope_name.encode()),
                                 zlib.crc32(scope_kind.encode()), zlib.crc32(group.encode()), int(k), int(trial)])
    return int(ss.generate_state(1, dtype=np.uint64)[0] % (2 ** 63))


def draw_weight_flips(rng: np.random.Generator, store: EncodedStore, layers: Sequence[int], bits: Sequence[int],
                      k: int) -> List[Tuple[int, int, int]]:
    """k DISTINCT elements uniformly from the layers in scope, one uniformly drawn bit of ``bits`` per element."""
    layers = list(layers)
    ns = store.n[layers]
    total = int(ns.sum())
    if k > total:
        raise ValueError(f"k={k} larger than the scope ({total} elements)")
    if not bits:
        raise ValueError("empty bit group")
    pick = rng.choice(total, size=k, replace=False)
    cum = np.cumsum(ns)
    li = np.searchsorted(cum, pick, side="right")
    elem = pick - (cum[li] - ns[li])
    b = rng.choice(np.asarray(bits, dtype=np.int64), size=k)
    return [(int(layers[l]), int(e), int(bb)) for l, e, bb in zip(li, elem, b)]


def draw_scale_flips(rng: np.random.Generator, store: EncodedStore, layers: Sequence[int], bits: Sequence[int],
                     k: int) -> List[Tuple[int, int]]:
    """k DISTINCT scale words (one per layer in per-tensor formats), one uniformly drawn float32 bit each."""
    cand = [l for l in layers if store.scales[l] is not None]
    if k > len(cand):
        raise ValueError(f"k={k} larger than the number of scale words in scope ({len(cand)})")
    pick = rng.choice(len(cand), size=k, replace=False)
    b = rng.choice(np.asarray(bits, dtype=np.int64), size=k)
    return [(int(cand[p]), int(bb)) for p, bb in zip(pick, b)]


# --------------------------------------------------------------------------------------------- model + evaluation
def build_model(model: nn.Module, store: EncodedStore, codec: S2.Codec, device) -> nn.Module:
    """Evaluation copy of ``model`` whose Conv/Linear weights are decoded from the stored bytes (weight-only)."""
    import copy
    m = copy.deepcopy(model).to(device).eval()
    params = dict(m.named_parameters())
    with torch.no_grad():
        for l, name in enumerate(store.names):
            params[name].copy_(torch.from_numpy(np.ascontiguousarray(decode_layer(codec, store, l))).to(device))
    return m


@torch.inference_mode()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 500, return_logits: bool = False) -> dict:
    """Accuracy / loss with the declared non-finite policy and NaN/Inf detection in activations and logits."""
    model.eval()
    mods = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))]
    flags = {n: torch.zeros((), dtype=torch.bool, device=X.device) for n, _ in mods}
    def make_hook(n):
        def hook(mod, inp, out):                         # must return None: a returned value would REPLACE the output
            flags[n].logical_or_(~torch.isfinite(out).all())
        return hook
    hooks = [m.register_forward_hook(make_hook(n)) for n, m in mods]
    try:
        logits = torch.cat([model(X[i:i + batch_size]).float() for i in range(0, X.shape[0], batch_size)])
    finally:
        for h in hooks:
            h.remove()
    rows_ok = torch.isfinite(logits).all(1)
    correct = rows_ok & (logits.argmax(1) == y)
    bad_layers = [n for n, f in flags.items() if bool(f)]
    loss = float(nn.functional.cross_entropy(logits, y)) if bool(rows_ok.all()) else None
    out = {"accuracy": 100.0 * float(correct.float().mean()), "correct": int(correct.sum()), "total": int(y.numel()),
           "loss": loss, "logit_rows_nonfinite": int((~rows_ok).sum()), "activation_nonfinite": bool(bad_layers),
           "first_nonfinite_activation_layer": bad_layers[0] if bad_layers else None}
    if return_logits:
        out["logits"] = logits
    return out


# --------------------------------------------------------------------------------------------- injector
class EncodedFaultInjector:
    def __init__(self, model: nn.Module, store: EncodedStore, codec: S2.Codec, device):
        self.store, self.codec, self.device = store, codec, torch.device(device)
        self.model = build_model(model, store, codec, device)
        params = dict(self.model.named_parameters())
        self.flat = []
        for name in store.names:
            p = params[name].data
            if not p.is_contiguous():
                raise ValueError(f"{name} is not contiguous")
            self.flat.append(p.view(-1))
        self._undo: List[tuple] = []

    def count_nonfinite_weights(self) -> int:
        return int(sum(int((~torch.isfinite(f)).sum()) for f in self.flat))

    def inject(self, weight_flips: Sequence[Tuple[int, int, int]] = (), scale_flips: Sequence[Tuple[int, int]] = ()) -> dict:
        """Flip bits of the STORED bytes: weight_flips = (layer, element, bit), scale_flips = (layer, float32 bit)."""
        if self._undo:
            raise RuntimeError("restore() must be called before the next inject()")
        by_layer: Dict[int, list] = {}
        for l, e, b in weight_flips:
            by_layer.setdefault(l, []).append((e, b))
        for l, items in by_layer.items():
            undo_bytes = [(l,) + self.store.flip(l, e, b) for e, b in items]     # may raise before anything else changed
            elems = np.unique(np.array([e for e, _ in items], dtype=np.int64))
            new = decode_elements(self.codec, self.store, l, elems)
            idx = torch.from_numpy(elems).to(self.device)
            old_vals = self.flat[l][idx].clone()
            self.flat[l][idx] = torch.from_numpy(new).to(self.device)
            self._undo.append(("w", l, undo_bytes, idx, old_vals))
        for l, b in scale_flips:
            old_scale = self.store.flip_scale(l, b)
            old_flat = self.flat[l].clone()
            with np.errstate(invalid="ignore", over="ignore"):
                full = decode_layer(self.codec, self.store, l)
            self.flat[l].copy_(torch.from_numpy(np.ascontiguousarray(full).reshape(-1)).to(self.device))
            self._undo.append(("s", l, old_scale, old_flat))
        return {"weight_flips": len(weight_flips), "scale_flips": len(scale_flips)}

    @torch.no_grad()
    def restore(self) -> None:
        for kind, l, a, b, *rest in reversed(self._undo):
            if kind == "w":
                for (layer, idx, old) in reversed(a):
                    self.store.words[layer][idx] = self.store.words[layer].dtype.type(old)
                self.flat[l][b] = rest[0]
            else:
                self.store.scales[l].reshape(1).view(np.uint32)[0] = np.uint32(a)       # exact bit pattern
                self.flat[l].copy_(b)
        self._undo = []


# --------------------------------------------------------------------------------------------- trial runner
REQUIRED_TRIAL_FIELDS = (
    "experiment_id", "run_id", "model", "dataset", "checkpoint_hash", "layer_id", "layer_name", "format", "codec_version",
    "fault_group", "bit_index", "fault_count", "trial_index", "seed", "sample_count", "clean_accuracy", "fault_accuracy",
    "clean_loss", "fault_loss", "nonfinite_weight_count", "output_flag", "encoded_hash_before", "encoded_hash_after_restore")


def run_trial(inj: EncodedFaultInjector, X, y, clean: dict, ctx: dict, *, fmt: str, scope_layers: Sequence[int],
              scope_name: str, scope_kind: str, group: str, k: int, trial: int, base_seed: int,
              explicit_weight_flips: Optional[Sequence[Tuple[int, int, int]]] = None,
              explicit_scale_flips: Optional[Sequence[Tuple[int, int]]] = None, return_logits: bool = False) -> dict:
    """One fault trial with the full Section 5.3 record. The injected state is ALWAYS restored (also on errors)."""
    store = inj.store
    seed = trial_seed(base_seed, fmt, scope_name, scope_kind, group, k, trial)
    rng = np.random.default_rng(seed)
    if explicit_weight_flips is not None or explicit_scale_flips is not None:
        wf, sf = list(explicit_weight_flips or []), list(explicit_scale_flips or [])
    elif k == 0:
        wf, sf = [], []
    elif scope_kind == "weights":
        wf, sf = draw_weight_flips(rng, store, scope_layers, group_bits(fmt, group), k), []
    else:
        wf, sf = [], draw_scale_flips(rng, store, scope_layers, group_bits(fmt, group, "scales"), k)
    # ---- scope check: every flip lies inside the declared scope and inside its layer
    scope = set(scope_layers)
    for l, e, b in wf:
        if l not in scope or not 0 <= e < int(store.n[l]):
            raise AssertionError(f"flip ({l},{e},{b}) outside the declared scope {scope_name}")
    for l, b in sf:
        if l not in scope or store.scales[l] is None:
            raise AssertionError(f"scale flip ({l},{b}) outside the declared scope {scope_name}")
    h0 = store.hash()
    ev = None
    nonfinite_w = 0
    h_fault = None
    try:
        inj.inject(wf, sf)
        h_fault = store.hash()
        nonfinite_w = inj.count_nonfinite_weights()
        ev = evaluate(inj.model, X, y, return_logits=return_logits)
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
    flips = []
    for l, e, b in wf:
        byte, pos = bit_location(fmt, e, b)
        flips.append({"target": "weight", "layer_id": l, "layer_name": store.names[l], "element": e, "bit_index": b,
                      "field": field_of_bit(fmt, b), "stored_index": byte, "bit_in_stored_unit": pos})
    for l, b in sf:
        flips.append({"target": "scale", "layer_id": l, "layer_name": store.names[l], "element": 0, "bit_index": b,
                      "field": field_of_bit(fmt, b, "scales"), "stored_index": 0, "bit_in_stored_unit": b})
    rec = {**ctx, "layer_id": scope_layers[0] if len(scope_layers) == 1 else -1,
           "layer_name": scope_name, "format": fmt, "scope_kind": scope_kind, "fault_group": group,
           "bit_index": [f["bit_index"] for f in flips], "fault_count": len(flips), "requested_k": k,
           "trial_index": trial, "seed": seed, "sample_count": int(y.numel()),
           "clean_accuracy": clean["accuracy"], "fault_accuracy": ev["accuracy"],
           "clean_loss": clean["loss"], "fault_loss": ev["loss"], "nonfinite_weight_count": nonfinite_w,
           "output_flag": ",".join(flags) if flags else "none", "failure": bool(flags),
           "encoded_hash_before": h0, "encoded_hash_faulted": h_fault, "encoded_hash_after_restore": h1,
           "restored_exactly": h0 == h1, "flip_applied_to_encoded_bytes": bool(flips) == (h_fault != h0),
           "flips": flips, "sampling": "distinct elements without replacement within the trial; bit uniform within the group"}
    if return_logits:
        rec["_logits"] = ev["logits"]
    return rec
