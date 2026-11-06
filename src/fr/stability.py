"""How trustworthy are the Fisher-based layer rankings?

Input: per-sample layer traces from ``sensitivity.per_sample_layer_traces`` (one pool per image source, e.g.
training images and held-out validation images). Everything here is a pure function of those arrays.

Questions answered:

1. *Sample size / subset stability.* Split a pool into disjoint chunks of n images; how similar are the layer
   rankings (Spearman of trace-per-parameter) between chunks, and how much does each layer's estimate vary?
2. *Concentration.* Is the trace carried by a handful of images? (effective sample size per layer, share of the
   top 1 % / 10 % of images, share carried by misclassified images)
3. *Selection stability.* Do chunks pick the same layers as the whole pool under the Fisher-only rule
   (lambda = 0, the same rank-normalised min-cost selection used in ``allocation``)?
4. *Source dependence.* Do training images and held-out images rank the layers alike?
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import allocation

DEFAULT_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)


def rankdata(a: Sequence[float]) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    for v in np.unique(a):
        m = a == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def weighted_jaccard(a: Sequence[str], b: Sequence[str], size_of: Dict[str, int]) -> float:
    """|A and B| / |A or B| counted in weights."""
    sa, sb = set(a), set(b)
    union = sum(size_of[n] for n in sa | sb)
    return sum(size_of[n] for n in sa & sb) / union if union else 1.0


def layer_metrics(traces: np.ndarray, counts: np.ndarray) -> Dict[str, np.ndarray]:
    total = traces.mean(axis=0)
    return {"total": total, "per_param": total / counts}


def _chunks(n_total: int, n: int):
    k = n_total // n
    return [slice(i * n, (i + 1) * n) for i in range(k)]


def _select_fisher_only(names, counts, per_param, target):
    costs = allocation.normalize(per_param, "rank")
    return allocation.select("min_cost", names, counts, target, costs=costs)["layers8"]


def _mean(x):
    x = [v for v in x if v == v]
    return float(np.mean(x)) if x else float("nan")


def analyze_pool(pool: Dict[str, object], sizes: Sequence[int] = DEFAULT_SIZES,
                 targets: Sequence[float] = (14.0, 12.0, 10.0), min_chunks: int = 4) -> dict:
    T, counts, names = pool["traces"], np.asarray(pool["counts"]), list(pool["names"])
    N, L = T.shape
    size_of = dict(zip(names, (int(c) for c in counts)))
    full = layer_metrics(T, counts)

    # 1. sample-size / subset stability
    size_rows = []
    for n in sizes:
        if n > N:
            continue
        ch = _chunks(N, n)
        pp = [layer_metrics(T[c], counts)["per_param"] for c in ch]
        row = {"n": n, "chunks": len(ch),
               "spearman_vs_full": _mean([spearman(p, full["per_param"]) for p in pp])}
        if len(ch) >= 2:
            pair = [spearman(a, b) for a, b in itertools.combinations(pp, 2)]
            row["spearman_between_chunks_mean"] = _mean(pair)
            row["spearman_between_chunks_min"] = float(np.nanmin(pair))
            est = np.vstack(pp)
            row["mean_cv_per_layer"] = float(np.mean(est.std(axis=0, ddof=1) / est.mean(axis=0)))
        size_rows.append(row)

    # 2. concentration
    ess = (T.sum(axis=0) ** 2) / (T ** 2).sum(axis=0)
    k1, k10 = max(1, int(np.ceil(0.01 * N))), max(1, int(np.ceil(0.10 * N)))
    srt = -np.sort(-T, axis=0)
    per_layer_top1 = srt[:k1].sum(axis=0) / T.sum(axis=0)
    t_all = T.sum(axis=1)
    srt_all = np.sort(t_all)[::-1]
    wrong = ~np.asarray(pool["correct"], dtype=bool)
    conc = {
        "n_samples": N, "n_misclassified": int(wrong.sum()),
        "ess_per_layer": {"min": float(ess.min()), "median": float(np.median(ess)), "max": float(ess.max())},
        "ess_over_n_median": float(np.median(ess) / N),
        "top1pct_share_per_layer": {"min": float(per_layer_top1.min()),
                                    "median": float(np.median(per_layer_top1)),
                                    "max": float(per_layer_top1.max())},
        "top1pct_share_all_layers": float(srt_all[:k1].sum() / t_all.sum()),
        "top10pct_share_all_layers": float(srt_all[:k10].sum() / t_all.sum()),
        "misclassified_share_of_trace": float(t_all[wrong].sum() / t_all.sum()),
        "spearman_trace_vs_nll": spearman(t_all, pool["nll"]),
        "mean_nll": float(np.mean(pool["nll"])),
    }

    # 3. selection stability (Fisher-only rule), using the largest n with at least `min_chunks` chunks
    n_ref = max([n for n in sizes if n <= N and N // n >= min_chunks], default=max(1, N // min_chunks))
    ch = _chunks(N, n_ref)
    sel_rows = []
    for tgt in targets:
        ref_sel = _select_fisher_only(names, counts, full["per_param"], tgt)
        sets = [_select_fisher_only(names, counts, layer_metrics(T[c], counts)["per_param"], tgt)
                for c in ch]
        sel_rows.append({
            "target": float(tgt), "chunk_size": n_ref, "chunks": len(ch), "full_pool_layers8": ref_sel,
            "jaccard_vs_full_mean": _mean([weighted_jaccard(s, ref_sel, size_of) for s in sets]),
            "jaccard_vs_full_min": float(min(weighted_jaccard(s, ref_sel, size_of) for s in sets)),
            "jaccard_between_chunks_mean": _mean([weighted_jaccard(a, b, size_of)
                                                  for a, b in itertools.combinations(sets, 2)]),
            "identical_to_full": int(sum(set(s) == set(ref_sel) for s in sets))})
    return {
        "layers": names,
        "full_pool": {"trace": full["total"].tolist(), "trace_per_param": full["per_param"].tolist()},
        "per_param_vs_total_spearman": spearman(full["per_param"], full["total"]),
        "sizes": size_rows, "concentration": conc, "selection": sel_rows,
    }


def compare_pools(a: Dict[str, object], b: Dict[str, object], targets: Sequence[float] = (14.0, 12.0, 10.0)) -> dict:
    """Source dependence: rankings from pool ``a`` (e.g. train) vs pool ``b`` (e.g. validation)."""
    names, counts = list(a["names"]), np.asarray(a["counts"])
    if names != list(b["names"]):
        raise ValueError("pools cover different layers")
    size_of = dict(zip(names, (int(c) for c in counts)))
    ma, mb = layer_metrics(a["traces"], counts), layer_metrics(b["traces"], counts)
    out = {"spearman_per_param": spearman(ma["per_param"], mb["per_param"]),
           "spearman_total": spearman(ma["total"], mb["total"]),
           "trace_ratio_b_over_a": {"median": float(np.median(mb["total"] / ma["total"])),
                                    "min": float((mb["total"] / ma["total"]).min()),
                                    "max": float((mb["total"] / ma["total"]).max())},
           "selection": []}
    for tgt in targets:
        sa = _select_fisher_only(names, counts, ma["per_param"], tgt)
        sb = _select_fisher_only(names, counts, mb["per_param"], tgt)
        out["selection"].append({"target": float(tgt), "jaccard": weighted_jaccard(sa, sb, size_of),
                                 "layers8_a": sa, "layers8_b": sb})
    return out


def _f(x, d=3):
    return "-" if x is None or x != x else f"{x:.{d}f}"


def render_markdown(results: Dict[str, dict], cross: Optional[dict] = None) -> str:
    lines = ["# Fisher stability", "",
             "Layer metric = Fisher trace per parameter (the quantity the lambda-sum and the Fisher-only rule rank).",
             "Spearman = rank correlation of the 21 layer values. Jaccard is counted in weights.", ""]
    for src, r in results.items():
        c = r["concentration"]
        lines += [f"## Source: {src} ({c['n_samples']} images, mean NLL {c['mean_nll']:.4f}, "
                  f"{c['n_misclassified']} misclassified)", "",
                  "### Ranking stability vs sample size", "",
                  "| n | chunks | Spearman between chunks (mean / min) | Spearman vs full pool | mean CV per layer |",
                  "|---|---|---|---|---|"]
        for s in r["sizes"]:
            lines.append(f"| {s['n']} | {s['chunks']} | {_f(s.get('spearman_between_chunks_mean'))} / "
                         f"{_f(s.get('spearman_between_chunks_min'))} | {_f(s['spearman_vs_full'])} | "
                         f"{_f(s.get('mean_cv_per_layer'))} |")
        e = c["ess_per_layer"]
        t1 = c["top1pct_share_per_layer"]
        lines += ["", "### Concentration", "",
                  f"* effective sample size per layer (of {c['n_samples']}): min {e['min']:.1f}, "
                  f"median {e['median']:.1f}, max {e['max']:.1f}",
                  f"* share of a layer's trace from the top 1% of images: min {t1['min']:.2f}, "
                  f"median {t1['median']:.2f}, max {t1['max']:.2f}",
                  f"* all layers: top 1% of images carry {c['top1pct_share_all_layers']:.2f}, "
                  f"top 10% carry {c['top10pct_share_all_layers']:.2f} of the total trace",
                  f"* misclassified images ({c['n_misclassified']}) carry {c['misclassified_share_of_trace']:.2f} "
                  f"of the total trace; Spearman(per-image trace, per-image NLL) = {_f(c['spearman_trace_vs_nll'])}",
                  f"* per-parameter vs total trace ranking: Spearman {_f(r['per_param_vs_total_spearman'])}", "",
                  "### Fisher-only selection stability", "",
                  "| target bits | chunk n | chunks | Jaccard vs full (mean / min) | Jaccard between chunks | "
                  "chunks identical to full |", "|---|---|---|---|---|---|"]
        for s in r["selection"]:
            lines.append(f"| {s['target']:g} | {s['chunk_size']} | {s['chunks']} | "
                         f"{_f(s['jaccard_vs_full_mean'])} / {_f(s['jaccard_vs_full_min'])} | "
                         f"{_f(s['jaccard_between_chunks_mean'])} | {s['identical_to_full']} |")
        lines.append("")
    if cross:
        lines += ["## Training images vs held-out images", "",
                  f"* Spearman of layer rankings: per parameter {_f(cross['spearman_per_param'])}, "
                  f"total {_f(cross['spearman_total'])}",
                  f"* held-out / training trace ratio per layer: median {cross['trace_ratio_b_over_a']['median']:.1f}, "
                  f"min {cross['trace_ratio_b_over_a']['min']:.1f}, max {cross['trace_ratio_b_over_a']['max']:.1f}", "",
                  "| target bits | Jaccard of Fisher-only selections |", "|---|---|"]
        for s in cross["selection"]:
            lines.append(f"| {s['target']:g} | {_f(s['jaccard'])} |")
        lines.append("")
    return "\n".join(lines)
