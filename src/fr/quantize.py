"""Post-training E4M11 quantization of a model (weights only).

Returns the dequantized FP32 model used for clean evaluation together with the
packed ``uint16`` buffers that a later fault-injection step will flip bits in.
"""
from __future__ import annotations

import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from . import e4m11

SCOPES = ("convlinear", "all")


def target_parameters(model: nn.Module, scope: str):
    """[(param_name, parameter)] to be stored as E4M11.

    convlinear : ``<module>.weight`` of every Conv2d/Linear (paper Sec 5.2 scope)
    all        : every entry of ``named_parameters()`` (old notebook: also BN
                 gamma/beta and the fc bias)
    """
    if scope == "convlinear":
        return [(f"{n}.weight", m.weight) for n, m in model.named_modules()
                if isinstance(m, (nn.Conv2d, nn.Linear))]
    if scope == "all":
        return list(model.named_parameters())
    raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")


def conv_linear_weights(model: nn.Module) -> "OrderedDict[str, np.ndarray]":
    """float32 numpy copies of all Conv2d/Linear weights, keyed ``<module>.weight``."""
    return OrderedDict((n, p.detach().cpu().numpy().copy())
                       for n, p in target_parameters(model, "convlinear"))


def all_parameters(model: nn.Module) -> "OrderedDict[str, np.ndarray]":
    return OrderedDict((n, p.detach().cpu().numpy().copy()) for n, p in model.named_parameters())


def quantize_model(model: nn.Module, mode: str = "paper", scope: str = "convlinear",
                   all_ones: str = "clamp"):
    """Deep-copy ``model`` and replace the selected parameters by E4M11 round-trips.

    Returns ``(quantized_model_on_cpu, buffers)`` where ``buffers`` maps the
    parameter name to ``{"words": uint16[N], "shape", "global_start", "count"}``.
    """
    q = copy.deepcopy(model).cpu().eval()
    buffers: "OrderedDict[str, dict]" = OrderedDict()
    start = 0
    with torch.no_grad():
        for name, p in target_parameters(q, scope):
            original = p.detach().cpu().numpy()
            words = e4m11.encode(original.reshape(-1), mode)
            deq = e4m11.decode(words, mode, all_ones).reshape(original.shape)
            p.copy_(torch.from_numpy(deq).to(p.dtype))
            buffers[name] = {"words": words, "shape": tuple(original.shape),
                             "global_start": start, "count": int(words.size)}
            start += words.size
    return q, buffers


def total_words(buffers) -> int:
    return sum(b["count"] for b in buffers.values())
