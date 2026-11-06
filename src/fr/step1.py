"""Step-1 pipelines used by scripts/eval_fp32.py and scripts/quantize_eval.py."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from . import data as fdata
from .config import Config
from .evaluate import evaluate
from .model import load_model
from .quantize import all_parameters, conv_linear_weights, quantize_model, total_words
from .report import compare_to_targets, load_targets, render_markdown
from .sqnr import sqnr_report, weight_stats
from .utils import dump_json, env_info, resolve_device

log = logging.getLogger("fr.step1")


def _loader(cfg: Config, norm, device):
    return fdata.get_test_loader(cfg.data_path(), norm, cfg.eval_batch_size, cfg.num_workers,
                                 cfg.test_subset, device.type == "cuda")


def _fp32_block(model, loader, device) -> dict:
    return {
        "metrics": evaluate(model, loader, device),
        "weight_stats": {
            "convlinear": weight_stats(list(conv_linear_weights(model).values())),
            "all_parameters": weight_stats(list(all_parameters(model).values())),
        },
    }


def eval_fp32(cfg: Config, checkpoint, diagnose_normalization: bool = False) -> dict:
    """FP32 accuracy + weight statistics (paper Table 3 style)."""
    device = resolve_device(cfg.device)
    model, ckpt = load_model(checkpoint)
    out = {"checkpoint": str(checkpoint), "device": str(device), "env": env_info(),
           "test_subset": cfg.test_subset,
           "weight_stats": {
               "convlinear": weight_stats(list(conv_linear_weights(model).values())),
               "all_parameters": weight_stats(list(all_parameters(model).values()))}}
    if diagnose_normalization:
        out["normalization_diagnostic"] = {}
        for name in ("none", "cifar10"):
            norm = fdata.resolve_normalization(name, "checkpoint", ckpt.get("config"))
            m = evaluate(model, _loader(cfg, norm, device), device)
            out["normalization_diagnostic"][name] = m
            log.info("normalization=%s -> %.2f%%", name, m["accuracy_percent"])
        out["note"] = ("Diagnostic only: choose the normalization from checkpoint['config'], "
                       "not by looking at test accuracy.")
    else:
        norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
        out["normalization"] = fdata.normalization_to_config(norm)
        out["metrics"] = evaluate(model, _loader(cfg, norm, device), device)
        log.info("FP32 accuracy %.2f%%", out["metrics"]["accuracy_percent"])
    dump_json(out, cfg.results_dir() / "fp32_report.json")
    return out


def quantize_eval(cfg: Config, checkpoint, modes: Sequence[str] = ("paper",),
                  scope: str = "convlinear", all_ones: str = "clamp",
                  bias: int = 7, targets_path: Optional[Path] = None) -> dict:
    device = resolve_device(cfg.device)
    model, ckpt = load_model(checkpoint)
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    loader = _loader(cfg, norm, device)
    targets = load_targets(targets_path) if targets_path else {}

    fp32 = _fp32_block(model, loader, device)
    log.info("FP32: %.2f%% (CE %.4f)", fp32["metrics"]["accuracy_percent"],
             fp32["metrics"]["cross_entropy"])
    orig_cl = conv_linear_weights(model)
    report = {
        "run_name": cfg.run_name, "checkpoint": str(checkpoint),
        "normalization": fdata.normalization_to_config(norm), "test_subset": cfg.test_subset,
        "test_total": fp32["metrics"]["total"], "device": str(device), "env": env_info(),
        "fp32": fp32, "e4m11": {}, "paper_targets_comparison": {},
    }
    for mode in modes:
        qmodel, buffers = quantize_model(model, mode=mode, scope=scope, all_ones=all_ones, bias=bias)
        metrics = evaluate(qmodel, loader, device)
        deq_cl = conv_linear_weights(qmodel)
        entry = {
            "mode": mode, "scope": scope, "all_ones": all_ones, "bias": bias,
            "n_tensors": len(buffers), "n_words": total_words(buffers),
            "metrics": metrics,
            "accuracy_drop_pp": fp32["metrics"]["accuracy_percent"] - metrics["accuracy_percent"],
            "cross_entropy_increase": metrics["cross_entropy"] - fp32["metrics"]["cross_entropy"],
            "sqnr": sqnr_report(orig_cl, deq_cl),   # paper Eq. 6: Conv+Linear weights
        }
        if scope == "all":
            orig_all, deq_all = all_parameters(model), all_parameters(qmodel)
            entry["sqnr_all_quantized_params"] = sqnr_report(orig_all, deq_all)["overall"]
        report["e4m11"][mode] = entry
        report["paper_targets_comparison"][mode] = compare_to_targets(
            fp32, entry, targets, cfg.test_subset) if targets else []
        log.info("E4M11[%s/%s]: %.2f%% (drop %+.2f pp), SQNR %.4f dB", mode, scope,
                 metrics["accuracy_percent"], entry["accuracy_drop_pp"],
                 entry["sqnr"]["overall"]["sqnr_db"])
    out_dir = cfg.results_dir()
    dump_json(report, out_dir / "step1_report.json")
    (out_dir / "step1_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report
