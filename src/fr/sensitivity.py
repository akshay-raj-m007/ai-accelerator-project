"""Diagonal Fisher information of the Conv2d/Linear weights (LAMPQ-style layer sensitivity).

Background (LAMPQ, Kim et al., AAAI 2026, Sec. "Type-aware Fisher-based Metric"): for a layer with weights
W_i the Hessian trace tr(H_i) predicts how much loss a given weight perturbation causes (Lemma 1, HAWQ-V2);
for a negative-log-likelihood loss the expected Hessian approximately equals the Fisher information F_i
(Lemma 2), so ``tr(F_i)`` is used as the sensitivity. Here we keep the whole *diagonal*:

    F_jj = E_x E_{y ~ p(y|x,theta)} [ (d log p(y|x,theta) / d w_j)^2 ]            ("true" Fisher)

estimated with one sampled label per calibration image. ``kind="empirical"`` uses the dataset label instead of
a sampled one (cheaper to interpret, but on a model that fits the training data its gradients are tiny and it
is not the quantity of Lemma 2). Both are provided and always labelled in the output.

Per-sample gradients are obtained with one backward pass per image (batch size 1, BatchNorm in eval mode), so
the squared gradient is exactly the per-sample one. Accumulation is in float64.

Only Conv2d/Linear ``.weight`` tensors are covered (the E4M11 buffer of the fault study). The model is not
modified. Calibration images must come from the training split (see ``data.get_calibration_loader``).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

KINDS = ("true", "empirical")


def weight_params(model: nn.Module):
    """[(param_name, parameter)] for every Conv2d/Linear weight, in definition order."""
    return [(f"{n}.weight", m.weight) for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


def iter_sample_grads(model: nn.Module, loader, device, kind: str = "true", seed: int = 0,
                      max_samples: Optional[int] = None):
    """Yield ``(grads, nll_true_label, correct)`` for every calibration sample, one backward pass each.

    ``grads`` are the gradients of ``-log p(y | x)`` w.r.t. the Conv2d/Linear weights (same order as
    ``weight_params``), with ``y`` sampled from the model (``kind="true"``) or the dataset label
    (``"empirical"``). ``nll_true_label`` / ``correct`` always refer to the dataset label. The model is put in
    eval mode and restored afterwards.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    was_training = model.training
    model.eval().to(device)
    params = [p for _, p in weight_params(model)]
    saved_flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    n = 0
    try:
        for images, labels in loader:
            for i in range(images.shape[0]):
                if max_samples is not None and n >= max_samples:
                    return
                x = images[i:i + 1].to(device)
                true_y = labels[i:i + 1].to(device)
                logits = model(x)
                logp = torch.log_softmax(logits.float(), dim=1)
                y = (torch.multinomial(logp.detach().exp(), 1, generator=gen).view(1)
                     if kind == "true" else true_y)
                loss = -logp.gather(1, y.view(1, 1)).sum()
                grads = torch.autograd.grad(loss, params)
                nll = float(-logp.detach().gather(1, true_y.view(1, 1)).sum())
                correct = bool(logits.argmax(1).item() == true_y.item())
                n += 1
                yield grads, nll, correct
    finally:
        for p, f in zip(params, saved_flags):
            p.requires_grad_(f)
        model.train(was_training)


def compute_fisher(model: nn.Module, loader, device, kind: str = "true", seed: int = 0,
                   max_samples: Optional[int] = None) -> Dict[str, object]:
    """Diagonal Fisher of every Conv2d/Linear weight.

    Returns ``{"fisher": {name: float32 tensor (cpu, weight shape)}, "n_samples", "kind", "seed",
    "mean_nll"}``. ``loader`` yields (images, labels); any batch size is accepted (samples are processed one
    at a time).
    """
    named = weight_params(model)
    acc = None
    n = 0
    nll_sum = 0.0
    for grads, nll, _ in iter_sample_grads(model, loader, device, kind, seed, max_samples):
        if acc is None:
            acc = [torch.zeros_like(g, dtype=torch.float64) for g in grads]
        for a, g in zip(acc, grads):
            a.add_(g.double().pow(2))
        nll_sum += nll
        n += 1
    if n == 0:
        raise ValueError("empty calibration loader")
    fisher = OrderedDict((name, (a / n).float().cpu()) for (name, _), a in zip(named, acc))
    return {"fisher": fisher, "n_samples": n, "kind": kind, "seed": int(seed),
            "mean_nll": nll_sum / n}


def per_sample_layer_traces(model: nn.Module, loader, device, kind: str = "true", seed: int = 0,
                            max_samples: Optional[int] = None) -> Dict[str, object]:
    """Per-sample Fisher trace of every layer: ``traces[i, l] = sum_j (d loss_i / d w_lj)^2``.

    Any subset of samples gives that subset's layer trace as the mean of its rows, which makes stability
    analyses (different sample sizes / disjoint subsets) free after one pass. Also returns the per-sample NLL
    and whether the model classified the sample correctly.
    """
    names = [n for n, _ in weight_params(model)]
    counts = [int(p.numel()) for _, p in weight_params(model)]
    rows, nlls, oks = [], [], []
    for grads, nll, correct in iter_sample_grads(model, loader, device, kind, seed, max_samples):
        rows.append([float(g.double().pow(2).sum()) for g in grads])
        nlls.append(nll)
        oks.append(correct)
    if not rows:
        raise ValueError("empty calibration loader")
    return {"names": names, "counts": np.asarray(counts, dtype=np.int64),
            "traces": np.asarray(rows, dtype=np.float64), "nll": np.asarray(nlls, dtype=np.float64),
            "correct": np.asarray(oks, dtype=bool), "kind": kind, "seed": int(seed)}


def layer_summary(fisher: Dict[str, torch.Tensor]) -> "OrderedDict[str, dict]":
    """Per layer: parameter count, Fisher trace (sum of the diagonal), trace per parameter."""
    out = OrderedDict()
    for name, f in fisher.items():
        f64 = f.double()
        tr = float(f64.sum())
        out[name] = {"count": int(f.numel()), "trace": tr, "trace_per_param": tr / f.numel(),
                     "max": float(f64.max())}
    return out


def save_fisher(result: Dict[str, object], path) -> None:
    torch.save({k: v for k, v in result.items()}, path)


def load_fisher(path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def fisher_numpy(fisher: Dict[str, torch.Tensor]) -> "OrderedDict[str, np.ndarray]":
    return OrderedDict((k, v.numpy().astype(np.float64)) for k, v in fisher.items())
