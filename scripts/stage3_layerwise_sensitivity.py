#!/usr/bin/env python
"""JFAMP (shared protocol) Stage 3: layer-wise quantization sensitivity, one layer at a time.

For each of the 53 Conv2d/Linear layers and each of the 7 primary formats (53 x 7 = 371 evaluations; --include-comparison adds the
labelled per-channel INT rows), only that layer is replaced by its DECODED frozen encoding (quantized_weights/<format>.npz of
stage2_quantize.py, content hashes verified against the manifest), everything else stays FP32, and the full model is evaluated on the
5,000 canonical validation images (batch 256). The layer is restored afterwards. Validation only; the test set is never read.

  python scripts/stage3_layerwise_sensitivity.py --checkpoint $CK --run-name mobilenetv2_cifar10 \
      --out-dir reports/mobilenetv2_cifar10/stage_03_layerwise_sensitivity

Writes layerwise_sensitivity_results.csv (one row per layer x format), layerwise_sensitivity_summary.csv (per format),
layer_type_sensitivity_summary.csv (per layer type x format), stage3_progress.json.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from fr import data as fdata
from fr import stage2 as S2
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, env_info, resolve_device, seed_everything

BATCH = 256
RESULT_FIELDS = ["layer_index", "layer_name", "layer_type", "num_weights", "format_key", "format_role", "accuracy_percent", "correct",
                 "accuracy_change_pp", "validation_loss", "loss_change", "changed_predictions", "prediction_agreement_percent",
                 "correct_to_wrong", "wrong_to_correct", "weight_mse", "weight_max_abs_error", "weight_sqnr_db", "payload_bytes", "scale_bytes",
                 "total_bytes"]


def layer_type(m: nn.Module) -> str:
    """depthwise: groups == in_channels; pointwise: 1x1 kernel; standard: other Conv2d; linear."""
    if isinstance(m, nn.Linear):
        return "linear"
    if m.groups == m.in_channels:
        return "depthwise"
    return "pointwise" if tuple(m.kernel_size) == (1, 1) else "standard"


@torch.inference_mode()
def logits_of(model, X):
    model.eval()
    return torch.cat([model(X[i:i + BATCH]).float() for i in range(0, X.shape[0], BATCH)])


def metrics(lg, lg0, y):
    p, p0 = lg.argmax(1), lg0.argmax(1)
    c, c0 = p == y, p0 == y
    n = int(y.numel())
    return {"accuracy_percent": 100.0 * c.float().mean().item(), "correct": int(c.sum()),
            "validation_loss": nn.functional.cross_entropy(lg, y).item(),
            "changed_predictions": int((p != p0).sum()), "prediction_agreement_percent": 100.0 * (1.0 - float((p != p0).sum()) / n),
            "correct_to_wrong": int((c0 & ~c).sum()), "wrong_to_correct": int((~c0 & c).sum())}


def summarise(rows, key_fn, label_fields):
    """Aggregate result rows by key_fn -> list of dicts (accuracy change statistics over layers)."""
    groups = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    out = []
    for k, rr in groups.items():
        ch = [r["accuracy_change_pp"] for r in rr]
        worst = min(rr, key=lambda r: r["accuracy_change_pp"])
        best = max(rr, key=lambda r: r["accuracy_change_pp"])
        d = dict(zip(label_fields, k if isinstance(k, tuple) else (k,)))
        d.update({"n_layers": len(rr), "mean_accuracy_change_pp": statistics.fmean(ch), "median_accuracy_change_pp": statistics.median(ch),
                  "worst_accuracy_change_pp": worst["accuracy_change_pp"], "worst_layer": worst["layer_name"],
                  "best_accuracy_change_pp": best["accuracy_change_pp"], "best_layer": best["layer_name"],
                  "layers_with_drop_gt_0p5pp": sum(1 for v in ch if v < -0.5), "layers_with_drop_gt_1pp": sum(1 for v in ch if v < -1.0),
                  "layers_with_improvement": sum(1 for v in ch if v > 0),
                  "mean_changed_predictions": statistics.fmean(r["changed_predictions"] for r in rr),
                  "mean_prediction_agreement_percent": statistics.fmean(r["prediction_agreement_percent"] for r in rr),
                  "mean_weight_sqnr_db": statistics.fmean(r["weight_sqnr_db"] for r in rr),
                  "total_payload_bytes": sum(r["payload_bytes"] for r in rr)})
        out.append(d)
    return out


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None, help="default results/<run>/jfamp/quantization")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--include-comparison", action="store_true", help="also evaluate the labelled comparison formats (per-channel INT, FTZ)")
    ap.add_argument("--eval-seed", type=int, default=42, help="seed recorded/set for the evaluation (no randomness is involved; the split seed comes from the checkpoint)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    seed_everything(args.eval_seed)
    qdir = Path(args.quant_dir) if args.quant_dir else cfg.results_dir() / "jfamp" / "quantization"
    out = Path(args.out_dir) if args.out_dir else cfg.results_dir() / "jfamp" / "stage_03_layerwise_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(args.checkpoint)
    man = json.load(open(qdir / "quantizer_manifest.json"))
    if man["checkpoint"]["sha256"] != sha256_file(ck):
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")

    model, ckpt = load_model(ck)
    model = model.to(device).eval()
    mods = quantizable_modules(model)
    names = [f"{n}.weight" for n, _ in mods]
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    loader = fdata.get_val_loader(cfg.data_path(), norm, BATCH, val_size, split_seed, cfg.num_workers, False, subset=None)
    X, y = load_tensors(loader, device)
    lg0 = logits_of(model, X)
    base = metrics(lg0, lg0, y)
    print(f"FP32 reference: {base['accuracy_percent']:.2f}% ({base['correct']}/{y.numel()}), loss {base['validation_loss']:.4f}")

    reg = S2.codec_registry()
    fmts = [k for k, f in man["formats"].items() if f["role"] == "primary" or args.include_comparison]
    encoded = {}
    for k in fmts:
        enc = S2.load_encoded(qdir / man["formats"][k]["encoded_file"], names)
        if S2.content_sha256(enc) != man["formats"][k]["content_sha256"]:
            sys.exit(f"{k}: encoded weights differ from the frozen content hash")
        encoded[k] = enc
    params = dict(model.named_parameters())
    original = {n: params[n].detach().clone() for n in names}
    rows, t0, total = [], time.time(), len(mods) * len(fmts)
    for li, ((mn, m), wn) in enumerate(zip(mods, names)):
        w = original[wn].cpu().numpy()
        for k in fmts:
            dec = reg[k].decode(encoded[k][wn])
            rt = S2.roundtrip_row(w, dec)
            st = S2.storage_row(reg[k], encoded[k][wn])
            with torch.no_grad():
                params[wn].copy_(torch.from_numpy(np.ascontiguousarray(dec)).to(device))
            r = metrics(logits_of(model, X), lg0, y)
            with torch.no_grad():
                params[wn].copy_(original[wn])                                   # restore FP32 before the next format
            rows.append({"layer_index": li, "layer_name": mn, "layer_type": layer_type(m), "num_weights": int(w.size), "format_key": k,
                         "format_role": man["formats"][k]["role"], "accuracy_percent": r["accuracy_percent"], "correct": r["correct"],
                         "accuracy_change_pp": r["accuracy_percent"] - base["accuracy_percent"], "validation_loss": r["validation_loss"],
                         "loss_change": r["validation_loss"] - base["validation_loss"], "changed_predictions": r["changed_predictions"],
                         "prediction_agreement_percent": r["prediction_agreement_percent"], "correct_to_wrong": r["correct_to_wrong"],
                         "wrong_to_correct": r["wrong_to_correct"], "weight_mse": rt["mse"], "weight_max_abs_error": rt["max_abs_error"],
                         "weight_sqnr_db": rt["sqnr_db"], "payload_bytes": st["payload_bytes"], "scale_bytes": st["scale_bytes"],
                         "total_bytes": st["total_bytes"]})
        print(f"[{len(rows):3d}/{total}] layer {li:2d} {mn:24s} {layer_type(m):9s} " + "  ".join(
            f"{k}:{rows[-len(fmts) + i]['accuracy_change_pp']:+.2f}" for i, k in enumerate(fmts)) + f"   ({time.time() - t0:.0f}s)", flush=True)
    # the model must be back to FP32 exactly
    assert all(torch.equal(params[n].detach(), original[n]) for n in names)
    assert torch.equal(logits_of(model, X), lg0), "restored model no longer reproduces the FP32 logits"

    write_csv(out / "layerwise_sensitivity_results.csv", rows, RESULT_FIELDS)
    write_csv(out / "layerwise_sensitivity_summary.csv", summarise(rows, lambda r: r["format_key"], ["format_key"]))
    write_csv(out / "layer_type_sensitivity_summary.csv",
              summarise(rows, lambda r: (r["layer_type"], r["format_key"]), ["layer_type", "format_key"]))
    dump_json({"stage": "JFAMP Stage 3 layer-wise quantization sensitivity", "status": "complete", "evaluations": len(rows), "expected_evaluations": total,
               "layers": len(mods), "formats": fmts, "seed": args.eval_seed, "batch_size": BATCH, "validation_images": int(y.numel()),
               "fp32_reference": base, "checkpoint_sha256": man["checkpoint"]["sha256"], "quantizer_frozen_config_sha256": man["frozen_config_sha256"],
               "int_scale_policy": man["int_scale_policy"], "evaluation": "FP32 compute on reconstructed weights; one layer quantized at a time; "
               "model restored to FP32 and re-verified bit-exact after the sweep", "seconds": time.time() - t0, "environment": env_info()},
              out / "stage3_progress.json")
    print("written:", out)


if __name__ == "__main__":
    main()
