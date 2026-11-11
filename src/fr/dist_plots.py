"""Per-layer weight and activation distributions (J-FAMP runbook Stage 1 profiling), 8 plots per page.

Weights: every value of every Conv2d/Linear weight tensor.
Activations: the INPUT of every Conv2d/Linear layer (what the stored weights are multiplied with) or, with
``point="output"``, its output (pre-BatchNorm for convolutions). Collected on a fixed set of images, in FP32, eval mode.
Exact statistics (count, mean, std, min, max, fraction of zeros) are accumulated over ALL values; the histogram and the
percentiles come from a uniform random subsample of at most ``max_samples`` values per layer (stated in each legend).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .model import quantizable_modules

PER_PAGE = 8
PCTS = (0.1, 99.9)


class ActStats:
    """Streaming exact statistics plus a bounded random subsample of the values."""

    def __init__(self, max_samples: int, gen: torch.Generator):
        self.n = 0
        self.s1 = 0.0
        self.s2 = 0.0
        self.mn = float("inf")
        self.mx = float("-inf")
        self.zeros = 0
        self.max_samples = max_samples
        self.gen = gen
        self.parts: List[torch.Tensor] = []
        self.shape: Optional[tuple] = None

    def add(self, t: torch.Tensor, per_batch: int) -> None:
        t = t.detach().float().reshape(-1)
        if self.shape is None:
            self.shape = tuple(t.shape)
        self.n += t.numel()
        d = t.double()
        self.s1 += float(d.sum())
        self.s2 += float((d * d).sum())
        self.mn = min(self.mn, float(t.min()))
        self.mx = max(self.mx, float(t.max()))
        self.zeros += int((t == 0).sum())
        if t.numel() > per_batch:
            idx = torch.randint(0, t.numel(), (per_batch,), generator=self.gen, device=t.device)
            t = t[idx]
        self.parts.append(t.cpu())

    def finish(self) -> dict:
        vals = torch.cat(self.parts).numpy()
        if vals.size > self.max_samples:
            vals = np.random.default_rng(0).choice(vals, self.max_samples, replace=False)
        mean = self.s1 / self.n
        std = math.sqrt(max(self.s2 / self.n - mean * mean, 0.0))
        return {"values": vals, "n": self.n, "mean": mean, "std": std, "min": self.mn, "max": self.mx,
                "zero_fraction": self.zeros / self.n, "sample_size": int(vals.size),
                "pcts": {p: float(np.percentile(vals, p)) for p in PCTS}}


def weight_stats(model: nn.Module) -> "Dict[str, dict]":
    out = {}
    for name, m in quantizable_modules(model):
        w = m.weight.detach().double().cpu().numpy().reshape(-1)
        out[name] = {"values": w, "n": int(w.size), "mean": float(w.mean()), "std": float(w.std()),
                     "min": float(w.min()), "max": float(w.max()), "zero_fraction": float((w == 0).mean()),
                     "sample_size": int(w.size), "pcts": {p: float(np.percentile(w, p)) for p in PCTS},
                     "shape": tuple(m.weight.shape)}
    return out


@torch.inference_mode()
def activation_stats(model: nn.Module, X: torch.Tensor, point: str = "input", batch_size: int = 128,
                     max_samples: int = 2_000_000, seed: int = 0) -> "Dict[str, dict]":
    """Run ``X`` through the model once and collect per-layer activation statistics."""
    if point not in ("input", "output"):
        raise ValueError("point must be 'input' or 'output'")
    model.eval()
    gen = torch.Generator(device=X.device).manual_seed(seed)
    mods = quantizable_modules(model)
    acc = {n: ActStats(max_samples, gen) for n, _ in mods}
    n_batches = math.ceil(X.shape[0] / batch_size)
    per_batch = max(max_samples // n_batches, 1000) * 2          # oversample, trimmed to max_samples in finish()
    hooks = []
    for name, m in mods:
        if point == "input":
            hooks.append(m.register_forward_pre_hook(lambda mod, inp, n=name: acc[n].add(inp[0], per_batch)))
        else:
            hooks.append(m.register_forward_hook(lambda mod, inp, out, n=name: acc[n].add(out, per_batch)))
    try:
        for i in range(0, X.shape[0], batch_size):
            model(X[i:i + batch_size])
    finally:
        for h in hooks:
            h.remove()
    res = {n: a.finish() for n, a in acc.items()}
    for n, m in mods:
        res[n]["shape"] = tuple(m.weight.shape)
    return res


def _group(name: str) -> str:
    return name.split(".")[0] if name.startswith("layer") else ("classifier" if name.startswith("fc") else "stem")


def plot_pages(stats: "Dict[str, dict]", kind: str, title: str, out_pdf, out_png_prefix, bins: int = 120,
               subtitle: str = "") -> List[str]:
    """Multi-page PDF (+ one PNG per page): 4 x 2 subplots per page, one heading per subplot, a legend in each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    names = list(stats)
    pages = math.ceil(len(names) / PER_PAGE)
    pngs = []
    xlabel = "weight value" if kind == "weights" else "activation value"
    with PdfPages(out_pdf) as pdf:
        for pg in range(pages):
            fig, axes = plt.subplots(4, 2, figsize=(12, 17))
            fig.suptitle(f"{title}\nlayers {pg * PER_PAGE + 1}-{min((pg + 1) * PER_PAGE, len(names))} of {len(names)}"
                         f"  (page {pg + 1}/{pages})" + (f"\n{subtitle}" if subtitle else ""), fontsize=13)
            for k, ax in enumerate(axes.ravel()):
                i = pg * PER_PAGE + k
                if i >= len(names):
                    ax.axis("off")
                    continue
                n = names[i]
                s = stats[n]
                v = s["values"]
                counts, edges = np.histogram(v, bins=bins)
                ax.stairs(counts, edges, fill=True, alpha=0.45, color="tab:blue",
                          label=f"histogram (n={s['n']:,}" + ("" if s["sample_size"] == s["n"] else f", {s['sample_size']:,} sampled") + ")")
                ax.set_yscale("log")
                ax.axvline(s["mean"], color="k", lw=1.2, label=f"mean {s['mean']:.4g}")
                ax.axvline(s["mean"] - s["std"], color="tab:green", lw=1, ls="--", label=f"mean ± std ({s['std']:.4g})")
                ax.axvline(s["mean"] + s["std"], color="tab:green", lw=1, ls="--")
                ax.axvline(s["pcts"][0.1], color="tab:orange", lw=1, ls=":", label=f"p0.1 / p99.9 ({s['pcts'][0.1]:.3g} / {s['pcts'][99.9]:.3g})")
                ax.axvline(s["pcts"][99.9], color="tab:orange", lw=1, ls=":")
                ax.plot([], [], " ", label=f"min {s['min']:.4g}   max {s['max']:.4g}")
                ax.plot([], [], " ", label=f"zeros {100 * s['zero_fraction']:.2f} %")
                ax.set_xlim(s["min"], s["max"]) if s["max"] > s["min"] else None
                ax.set_title(f"L{i}  {n}  [{_group(n)}]  layer weight shape {tuple(s["shape"])}", fontsize=10, fontweight="bold")
                ax.set_xlabel(xlabel, fontsize=8)
                ax.set_ylabel("count (log scale)", fontsize=8)
                ax.tick_params(labelsize=8)
                ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
                ax.grid(alpha=0.25)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig)
            png = f"{out_png_prefix}_page{pg + 1}.png"
            fig.savefig(png, dpi=130)
            pngs.append(png)
            plt.close(fig)
    return pngs
