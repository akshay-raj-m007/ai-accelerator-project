"""Mixed-format post-training quantization: each Conv2d/Linear weight tensor is stored either as E4M11
(16-bit words, the fixed baseline: ``subnormal`` mode, bias 7) or as one of the 8-bit formats of
``formats8`` (8-bit words plus scale metadata).

``quantize_mixed`` returns the dequantized model for clean evaluation and, per tensor, the packed buffer a
later fault-injection step flips bits in:

    buffers[param_name] = {"format", "bits", "words" (uint16 or uint8), "scale", "shape",
                           "global_start" (offset in weights), "count", "metadata_bits"}

Fault groups, the global index space and the sampling model for a mixed buffer are decided in the fault
step, not here.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from . import e4m11, formats8
from .quantize import target_parameters

BASE = "e4m11"
ALL_FORMATS = (BASE,) + formats8.FORMATS8


def quantize_tensor(w: np.ndarray, fmt: str, e4m11_mode: str = "subnormal", bias: int = e4m11.BIAS,
                    all_ones: str = "clamp") -> dict:
    """One tensor in format ``fmt`` -> dict with words, scale, dequantized, bits, metadata_bits."""
    if fmt == BASE:
        words = e4m11.encode(w.reshape(-1), e4m11_mode, bias)
        deq = e4m11.decode(words, e4m11_mode, all_ones, bias).reshape(w.shape)
        return {"format": BASE, "bits": 16, "words": words, "scale": None, "shape": tuple(w.shape),
                "dequantized": deq, "metadata_bits": 0}
    return formats8.quantize8(w, fmt)


def quantize_mixed(model: nn.Module, assignment: Mapping[str, str], e4m11_mode: str = "subnormal",
                   bias: int = e4m11.BIAS, all_ones: str = "clamp",
                   cache: Optional[dict] = None):
    """Deep-copy ``model``; store every Conv2d/Linear weight in ``assignment[name]`` (default E4M11).

    ``assignment`` maps parameter names (``layer1.0.conv1.weight``) to a format in ``ALL_FORMATS``; missing
    names are E4M11. ``cache`` (a dict) memoises per-(name, format) results across calls.
    """
    q = copy.deepcopy(model).cpu().eval()
    params = target_parameters(q, "convlinear")
    unknown = set(assignment) - {n for n, _ in params}
    if unknown:
        raise KeyError(f"unknown layers in assignment: {sorted(unknown)}")
    buffers: "OrderedDict[str, dict]" = OrderedDict()
    start = 0
    with torch.no_grad():
        for name, p in params:
            fmt = assignment.get(name, BASE)
            if fmt not in ALL_FORMATS:
                raise ValueError(f"unknown format {fmt!r} for {name}")
            key = (name, fmt, e4m11_mode, bias, all_ones)
            if cache is not None and key in cache:
                r = cache[key]
            else:
                r = quantize_tensor(p.detach().cpu().numpy(), fmt, e4m11_mode, bias, all_ones)
                if cache is not None:
                    cache[key] = r
            p.copy_(torch.from_numpy(r["dequantized"]).to(p.dtype))
            buffers[name] = {"format": r["format"], "bits": r["bits"], "words": r["words"],
                             "scale": r["scale"], "shape": r["shape"], "global_start": start,
                             "count": int(r["words"].size), "metadata_bits": r["metadata_bits"]}
            start += int(r["words"].size)
    return q, buffers


def bit_accounting(buffers: Mapping[str, dict]) -> dict:
    """Storage cost of a mixed buffer: payload bits per weight (the size target) and with scale metadata."""
    n = sum(b["count"] for b in buffers.values())
    payload = sum(b["count"] * b["bits"] for b in buffers.values())
    meta = sum(b["metadata_bits"] for b in buffers.values())
    by_fmt: Dict[str, int] = {}
    for b in buffers.values():
        by_fmt[b["format"]] = by_fmt.get(b["format"], 0) + b["count"]
    return {"n_weights": n, "payload_bits": payload, "metadata_bits": meta,
            "avg_bits_payload": payload / n, "avg_bits_with_metadata": (payload + meta) / n,
            "weights_by_format": by_fmt}


def dequantized_weights(buffers: Mapping[str, dict], e4m11_mode: str = "subnormal",
                        bias: int = e4m11.BIAS, all_ones: str = "clamp") -> "OrderedDict[str, np.ndarray]":
    """Decode the stored words back to float32 (what a fault-free read returns)."""
    out = OrderedDict()
    for name, b in buffers.items():
        if b["format"] == BASE:
            out[name] = e4m11.decode(b["words"], e4m11_mode, all_ones, bias).reshape(b["shape"])
        else:
            out[name] = formats8.dequantize8(b["words"], b["scale"], b["shape"], b["format"], all_ones)
    return out
