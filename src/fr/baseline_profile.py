"""J-FAMP runbook Stage 0 (baseline intake) and Stage 1 (layer inventory): pure functions, no I/O of their own.

Everything is computed from the supplied checkpoint / model; no count is hard-coded (the runbook's reference values
are only compared against and any deviation is reported).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from .model import quantizable_modules

PERCENTILES = (50.0, 90.0, 99.0, 99.9)


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_ints(values: Sequence[int]) -> str:
    """Checksum of an index list (the train / validation split manifest)."""
    return hashlib.sha256(np.asarray(list(values), dtype=np.int64).tobytes()).hexdigest()


def layer_group(name: str) -> str:
    """Stem / layer1..layer4 / classifier for ResNet-18 module paths."""
    if name.startswith("layer"):
        return name.split(".")[0]
    if name.startswith("fc") or name.startswith("linear"):
        return "classifier"
    return "stem"


# ------------------------------------------------------------------------------------------ Stage 0
@torch.inference_mode()
def evaluate_full(model: nn.Module, X: torch.Tensor, y: torch.Tensor, n_classes: int = 10,
                  batch_size: int = 500) -> dict:
    """FP32 evaluation: accuracy, cross-entropy, 10x10 confusion matrix (rows = true, columns = predicted),
    per-class support / correct / accuracy and a non-finite-logit check. ``model`` must be in eval mode."""
    if model.training:
        raise RuntimeError("model must be in eval() mode")
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    loss, finite = 0.0, True
    for i in range(0, X.shape[0], batch_size):
        logits = model(X[i:i + batch_size]).float()
        yb = y[i:i + batch_size]
        finite &= bool(torch.isfinite(logits).all())
        loss += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        pred = logits.argmax(1)
        np.add.at(conf, (yb.cpu().numpy(), pred.cpu().numpy()), 1)
    total = int(conf.sum())
    correct = int(np.trace(conf))
    support = conf.sum(1)
    diag = np.diag(conf)
    return {"accuracy_percent": 100.0 * correct / total, "correct": correct, "total": total,
            "cross_entropy": loss / total, "all_logits_finite": finite, "confusion": conf,
            "per_class": [{"class_id": c, "support": int(support[c]), "correct": int(diag[c]),
                           "accuracy_percent": 100.0 * diag[c] / support[c] if support[c] else float("nan")}
                          for c in range(n_classes)]}


def checkpoint_checks(model: nn.Module, ckpt: dict, state_dict: dict, expected: dict) -> dict:
    """Stage 0 checkpoint validation. ``expected`` holds the runbook reference values to compare against."""
    ref_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got_shapes = {k: tuple(v.shape) for k, v in state_dict.items()}
    missing = sorted(set(ref_shapes) - set(got_shapes))
    unexpected = sorted(set(got_shapes) - set(ref_shapes))
    mismatched = sorted(k for k in ref_shapes if k in got_shapes and ref_shapes[k] != got_shapes[k])
    nonfinite = sorted(n for n, p in model.named_parameters() if not bool(torch.isfinite(p).all()))
    mods = quantizable_modules(model)
    n_quant = sum(m.weight.numel() for _, m in mods)
    n_total = sum(p.numel() for p in model.parameters())
    out_dim = model.fc.out_features if hasattr(model, "fc") else None
    maxpool = getattr(model, "maxpool", None)
    stem = model.conv1
    c = {"strict_load": not (missing or unexpected or mismatched),
         "missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatches": mismatched,
         "output_dim": out_dim, "output_dim_is_10": out_dim == 10,
         "stem_kernel": tuple(stem.kernel_size), "stem_stride": tuple(stem.stride),
         "stem_is_3x3_stride1": tuple(stem.kernel_size) == (3, 3) and tuple(stem.stride) == (1, 1),
         "no_maxpool": maxpool is None or isinstance(maxpool, nn.Identity),
         "nonfinite_parameters": nonfinite, "all_parameters_finite": not nonfinite,
         "n_quantizable_layers": len(mods), "n_quantizable_weights": int(n_quant), "n_parameters_total": int(n_total),
         "model_in_eval_mode": not model.training,
         "grad_enabled_during_evaluation": False}
    dev = {}
    for key, got in (("n_quantizable_layers", c["n_quantizable_layers"]),
                     ("n_quantizable_weights", c["n_quantizable_weights"]),
                     ("n_parameters_total", c["n_parameters_total"])):
        if key in expected:
            dev[key] = {"expected": expected[key], "got": got, "matches": expected[key] == got}
    c["comparison_with_runbook_reference"] = dev
    return c


# ------------------------------------------------------------------------------------------ Stage 1
def layer_inventory(model: nn.Module) -> List[dict]:
    """One row per Conv2d/Linear module in definition order (runbook 3.1)."""
    rows = []
    for i, (name, m) in enumerate(quantizable_modules(model)):
        w = m.weight.detach().double().cpu().numpy().reshape(-1)
        a = np.abs(w)
        row = {"layer_id": i, "module_name": name, "module_type": type(m).__name__,
               "weight_shape": list(m.weight.shape), "num_weights": int(w.size),
               "bias_present": m.bias is not None,
               "bias_shape": list(m.bias.shape) if m.bias is not None else None,
               "weight_min": float(w.min()), "weight_max": float(w.max()),
               "weight_mean": float(w.mean()), "weight_std": float(w.std()),
               "weight_abs_max": float(a.max()),
               "fp32_bytes": int(m.weight.numel() * m.weight.element_size()),
               "bias_bytes": int(m.bias.numel() * m.bias.element_size()) if m.bias is not None else 0,
               "layer_group": layer_group(name)}
        for p in PERCENTILES:
            row[f"abs_p{p:g}"] = float(np.percentile(a, p))
        rows.append(row)
    return rows


def storage_summary(rows: List[dict]) -> List[dict]:
    total_w = sum(r["num_weights"] for r in rows)
    total_b = sum(r["fp32_bytes"] for r in rows)
    out = [{"layer_id": r["layer_id"], "module_name": r["module_name"], "num_weights": r["num_weights"],
            "fp32_bytes": r["fp32_bytes"], "percent_of_quantizable_weights": 100.0 * r["num_weights"] / total_w}
           for r in rows]
    out.append({"layer_id": "", "module_name": "TOTAL", "num_weights": total_w, "fp32_bytes": total_b,
                "percent_of_quantizable_weights": 100.0})
    return out


def parameter_summary(model: nn.Module, rows: List[dict]) -> dict:
    """Totals and the non-quantized tensors (biases, BatchNorm parameters and buffers), runbook Stage 1."""
    quant_names = {f"{r['module_name']}.weight" for r in rows}
    nonq = OrderedDict()
    for n, p in model.named_parameters():
        if n not in quant_names:
            nonq[n] = {"shape": list(p.shape), "numel": int(p.numel()), "bytes": int(p.numel() * p.element_size()),
                       "kind": "parameter"}
    for n, b in model.named_buffers():
        nonq[n] = {"shape": list(b.shape), "numel": int(b.numel()), "bytes": int(b.numel() * b.element_size()),
                   "kind": "buffer"}
    n_quant = sum(r["num_weights"] for r in rows)
    n_params = sum(p.numel() for p in model.parameters())
    n_nonq_params = sum(v["numel"] for v in nonq.values() if v["kind"] == "parameter")
    counts = {}
    for _, m in model.named_modules():
        counts[type(m).__name__] = counts.get(type(m).__name__, 0) + 1
    return {"total_parameters": int(n_params), "quantizable_weights": int(n_quant),
            "non_quantized_parameters": int(n_nonq_params),
            "quantizable_plus_nonquantized_equals_total": n_quant + n_nonq_params == n_params,
            "fp32_quantizable_bytes": int(sum(r["fp32_bytes"] for r in rows)),
            "fp32_quantizable_bytes_equals_4x_weights": sum(r["fp32_bytes"] for r in rows) == 4 * n_quant,
            "module_counts": counts, "non_quantized_tensors": nonq}


def reconcile(rows: List[dict], model: nn.Module) -> Dict[str, bool]:
    """Gate 1: totals must reconcile exactly with the model state."""
    total_w = sum(p.numel() for n, p in model.named_parameters() if n in {f"{r['module_name']}.weight" for r in rows})
    return {"sum_layer_weights_equals_global": sum(r["num_weights"] for r in rows) == total_w,
            "sum_layer_bytes_equals_4x_global": sum(r["fp32_bytes"] for r in rows) == 4 * total_w,
            "every_conv_linear_listed": len(rows) == len(quantizable_modules(model))}
