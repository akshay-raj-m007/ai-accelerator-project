#!/usr/bin/env python
"""J-FAMP runbook Stage 2: candidate quantization and clean evaluation (INT3 deferred).

Formats (see src/fr/stage2.py): E4M11-SN, FP16 (E5M10), FP8 E4M3FN, INT8 per-tensor, INT4 per-tensor (packed), and
E4M11-FTZ as a labelled comparison. Each is generated independently from the ORIGINAL FP32 checkpoint and stored as real
encoded words; evaluation decodes the stored words (weight-only: biases, BatchNorm and activations stay FP32).

Two stages, so that the test set cannot influence the frozen configuration (runbook 4.3):

  --stage quantize   encode every format, round-trip metrics, storage, VALIDATION accuracy/loss (all 5,000 images),
                     codec self-tests, quantizer_manifest.json with the frozen-configuration hash. No test image is read.
  --stage test       verifies the frozen configuration (content hashes of the stored encoded weights) and then evaluates
                     the official test set for the same formats; adds the test columns to clean_accuracy.csv.

  python scripts/stage2_quantize.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch --stage quantize
  python scripts/stage2_quantize.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch --stage test

Outputs under results/<run>/jfamp/quantization/:
  quantizer_manifest.json  quantization_metrics.csv  clean_accuracy.csv  storage_summary.csv  quantized_weights/<format>.npz
(quantized_weights/ is regenerable from the checkpoint and the frozen configuration; its content hashes are in the manifest.)
"""
import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import logging
import math
import os
import stat
import sys
from collections import OrderedDict
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from fr import data as fdata
from fr import stage2 as S2
from fr.baseline_profile import evaluate_full, layer_group, sha256_file
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, env_info, git_info, resolve_device

SRC_FILES = ["src/fr/e4m11.py", "src/fr/formats8.py", "src/fr/stage2.py", "src/fr/sqnr.py", "scripts/stage2_quantize.py"]
ROOT = Path(__file__).resolve().parents[1]


def fnum(x):
    return "inf" if x == math.inf else "-inf" if x == -math.inf else x


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fnum(r.get(k, "")) for k in fields})


def model_with_weights(model, decoded):
    m = copy.deepcopy(model).eval()
    params = dict(m.named_parameters())
    with torch.no_grad():
        for name, w in decoded.items():
            params[name].copy_(torch.from_numpy(np.ascontiguousarray(w)).to(params[name].dtype))
    return m


def eval_split(model, X, y, device):
    with torch.inference_mode():
        r = evaluate_full(model.to(device).eval(), X, y)
    return {"accuracy_percent": r["accuracy_percent"], "correct": r["correct"], "total": r["total"],
            "cross_entropy": r["cross_entropy"], "all_logits_finite": r["all_logits_finite"]}


def source_info():
    files = {}
    for f in SRC_FILES:
        p = ROOT / f
        files[f] = sha256_file(p) if p.exists() else None
    return {"git": git_info(), "source_sha256": files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stage", choices=["quantize", "test"], default="quantize")
    ap.add_argument("--out-dir", default=None, help="default results/<run>/jfamp/quantization")
    ap.add_argument("--no-comparison", action="store_true", help="skip the E4M11-FTZ comparison row")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    out = Path(args.out_dir) if args.out_dir else cfg.results_dir() / "jfamp" / "quantization"
    qdir = out / "quantized_weights"
    out.mkdir(parents=True, exist_ok=True)
    qdir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "quantizer_manifest.json"
    ck = Path(args.checkpoint)

    model, ckpt = load_model(ck)
    model.eval()
    mods = quantizable_modules(model)
    layer_names = [f"{n}.weight" for n, _ in mods]
    weights = OrderedDict((f"{n}.weight", m.weight.detach().cpu().numpy().astype(np.float32)) for n, m in mods)
    ck_sha = sha256_file(ck)
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    codecs = S2.make_codecs(not args.no_comparison)

    # ============================================================================================ test stage
    if args.stage == "test":
        if not manifest_path.exists():
            sys.exit("no quantizer_manifest.json: run --stage quantize first (the configuration must be frozen before the test set is used)")
        codecs = S2.make_codecs(True)                      # the manifest decides which formats are evaluated
        man = json.load(open(manifest_path))
        if man["checkpoint"]["sha256"] != ck_sha:
            sys.exit("checkpoint SHA-256 differs from the one the configuration was frozen on")
        test_loader = fdata.get_test_loader(cfg.data_path(), norm, cfg.eval_batch_size, cfg.num_workers, None, False)
        Xt, yt = load_tensors(test_loader, device)
        rows_csv = list(csv.DictReader(open(out / "clean_accuracy.csv")))
        by_key = {r["format_key"]: r for r in rows_csv}
        t_fp32 = eval_split(model, Xt, yt, device)
        by_key["fp32"].update(test_accuracy_percent=t_fp32["accuracy_percent"], test_correct=t_fp32["correct"],
                              test_total=t_fp32["total"], test_cross_entropy=t_fp32["cross_entropy"],
                              test_drop_pp=0.0, test_delta_ce=0.0)
        for key, fm in man["formats"].items():
            enc = S2.load_encoded(out / fm["encoded_file"], layer_names)
            if S2.content_sha256(enc) != fm["content_sha256"]:
                sys.exit(f"{key}: encoded weights differ from the frozen content hash; refusing to evaluate")
            dec = OrderedDict((n, codecs[key].decode(e)) for n, e in enc.items())
            r = eval_split(model_with_weights(model, dec), Xt, yt, device)
            by_key[key].update(test_accuracy_percent=r["accuracy_percent"], test_correct=r["correct"], test_total=r["total"],
                               test_cross_entropy=r["cross_entropy"], test_drop_pp=t_fp32["accuracy_percent"] - r["accuracy_percent"],
                               test_delta_ce=r["cross_entropy"] - t_fp32["cross_entropy"])
            print(f"{key:12s} test {r['accuracy_percent']:.2f} %  ({r['correct']}/{r['total']})  drop {t_fp32['accuracy_percent'] - r['accuracy_percent']:+.2f} pp")
        fields = list(rows_csv[0].keys())
        for f in ("test_accuracy_percent", "test_correct", "test_total", "test_cross_entropy", "test_drop_pp", "test_delta_ce"):
            if f not in fields:
                fields.append(f)
        write_csv(out / "clean_accuracy.csv", list(by_key.values()), fields)
        man["test_stage"] = {"evaluated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                             "frozen_config_sha256_verified": man["frozen_config_sha256"], "test_images": int(yt.numel())}
        dump_json(man, manifest_path)
        print("test columns added to", out / "clean_accuracy.csv")
        return

    # ============================================================================================ quantize stage
    tests = S2.run_codec_selftests()
    bad = [t for t in tests if not t["passed"]]
    print(f"codec self-tests: {len(tests) - len(bad)}/{len(tests)} passed")
    for t in bad:
        print("  FAILED:", t["test"], t["detail"])
    if bad:
        sys.exit("codec self-tests failed; nothing was written")

    val_loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed,
                                      cfg.num_workers, False, subset=None)
    Xv, yv = load_tensors(val_loader, device)
    v_fp32 = eval_split(model, Xv, yv, device)
    n_total_w = sum(w.size for w in weights.values())
    fp32_weight_bytes = 4 * n_total_w
    nonq_param_bytes = sum(p.numel() * p.element_size() for n, p in model.named_parameters() if n not in weights)
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    fp32_full_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + buffer_bytes

    metric_rows, storage_rows, acc_rows, fmt_manifest = [], [], [], OrderedDict()
    acc_rows.append({"format_key": "fp32", "variant": "FP32 (IEEE 754 single, reference)", "role": "reference",
                     "val_accuracy_percent": v_fp32["accuracy_percent"], "val_correct": v_fp32["correct"],
                     "val_total": v_fp32["total"], "val_cross_entropy": v_fp32["cross_entropy"], "val_drop_pp": 0.0,
                     "val_delta_ce": 0.0})
    storage_rows.append({"format_key": "fp32", "variant": "FP32 reference", "role": "reference", "num_weights": n_total_w,
                         "theoretical_bits": 32 * n_total_w, "theoretical_bits_per_weight": 32.0,
                         "payload_bytes": fp32_weight_bytes, "padding_bits": 0, "scale_bytes": 0, "codec_metadata_bytes": 0,
                         "total_bytes": fp32_weight_bytes, "actual_bits_per_weight": 32.0,
                         "compression_ratio_weights": 1.0, "compression_ratio_payload_only": 1.0,
                         "non_quantized_param_bytes_fp32": nonq_param_bytes, "buffer_bytes": buffer_bytes,
                         "full_model_bytes": fp32_full_bytes, "compression_ratio_full_model": 1.0})

    for key, codec in codecs.items():
        encoded, parts = OrderedDict(), []
        dec = OrderedDict()
        tot = {"theoretical_bits": 0, "payload_bytes": 0, "padding_bits": 0, "scale_bytes": 0, "codec_metadata_bytes": 0,
               "total_bytes": 0}
        for lid, (name, w) in enumerate(weights.items()):
            enc = codec.encode(w)                                     # always from the original FP32 tensor
            encoded[name] = enc
            what = codec.decode(enc)                                  # decoded from the stored words only
            dec[name] = what
            rt = S2.roundtrip_row(w, what)
            st = S2.storage_row(codec, enc)
            parts.append(rt)
            for k in tot:
                tot[k] += st[k]
            metric_rows.append({"format_key": key, "variant": codec.label, "layer_id": lid, "module_name": name[:-7],
                                "layer_group": layer_group(name), "num_weights": enc["n"], **rt, "n_clipped": enc["n_clipped"],
                                "payload_bytes": st["payload_bytes"], "scale_bytes": st["scale_bytes"],
                                "total_bytes": st["total_bytes"], "theoretical_bits": st["theoretical_bits"]})
        g = S2.pooled_row(parts)
        metric_rows.append({"format_key": key, "variant": codec.label, "layer_id": "", "module_name": "GLOBAL",
                            "layer_group": "", "num_weights": g["n"], **g, "n_clipped": sum(r["n_clipped"] for r in metric_rows if r["format_key"] == key and r["module_name"] != "GLOBAL"),
                            "payload_bytes": tot["payload_bytes"], "scale_bytes": tot["scale_bytes"],
                            "total_bytes": tot["total_bytes"], "theoretical_bits": tot["theoretical_bits"]})
        # ---- storage
        full = tot["total_bytes"] + nonq_param_bytes + buffer_bytes
        storage_rows.append({"format_key": key, "variant": codec.label, "role": codec.role, "num_weights": n_total_w,
                             **tot, "theoretical_bits_per_weight": tot["theoretical_bits"] / n_total_w,
                             "actual_bits_per_weight": 8 * tot["total_bytes"] / n_total_w,
                             "compression_ratio_weights": fp32_weight_bytes / tot["total_bytes"],
                             "compression_ratio_payload_only": fp32_weight_bytes / tot["payload_bytes"],
                             "non_quantized_param_bytes_fp32": nonq_param_bytes, "buffer_bytes": buffer_bytes,
                             "full_model_bytes": full, "compression_ratio_full_model": fp32_full_bytes / full})
        # ---- save encoded weights (immutable), content hash
        path = qdir / f"{key}.npz"
        chash = S2.content_sha256(encoded)
        if path.exists():
            old = S2.load_encoded(path, layer_names)
            if S2.content_sha256(old) != chash:
                sys.exit(f"{path} exists with different content; the encoded weights are immutable. Use a new --out-dir.")
        else:
            S2.save_encoded(path, encoded)
            os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        # ---- validation evaluation of the DECODED stored weights
        r = eval_split(model_with_weights(model, dec), Xv, yv, device)
        acc_rows.append({"format_key": key, "variant": codec.label, "role": codec.role,
                         "val_accuracy_percent": r["accuracy_percent"], "val_correct": r["correct"], "val_total": r["total"],
                         "val_cross_entropy": r["cross_entropy"], "val_drop_pp": v_fp32["accuracy_percent"] - r["accuracy_percent"],
                         "val_delta_ce": r["cross_entropy"] - v_fp32["cross_entropy"]})
        fmt_manifest[key] = {"role": codec.role, "config": codec.config(), "encoded_file": f"quantized_weights/{key}.npz",
                             "content_sha256": chash, "encoded_file_bytes": path.stat().st_size,
                             "stored_total_bytes": tot["total_bytes"],
                             "quantization_simulated_or_packed": "stored in the actual encoded representation "
                                                                 "(uint16 / uint8 words, INT4 packed two per byte); evaluation decodes them to FP32",
                             "global_sqnr_db": g["sqnr_db"], "global_rmse": g["rmse"],
                             "validation": {k: acc_rows[-1][k] for k in ("val_accuracy_percent", "val_correct", "val_total", "val_cross_entropy")}}
        print(f"{key:12s} {codec.label:42s} val {r['accuracy_percent']:6.2f} %  drop {acc_rows[-1]['val_drop_pp']:+6.2f} pp  "
              f"SQNR {g['sqnr_db']:7.2f} dB  {tot['total_bytes']:>10,d} B ({fp32_weight_bytes / tot['total_bytes']:.2f}x)")

    mcols = ["format_key", "variant", "layer_id", "module_name", "layer_group", "num_weights", "mse", "rmse", "mae",
             "max_abs_error", "relative_l2", "sqnr_db", "n_clipped", "zeroed_from_nonzero", "decoded_nonfinite",
             "payload_bytes", "scale_bytes", "total_bytes", "theoretical_bits"]
    write_csv(out / "quantization_metrics.csv", metric_rows, mcols)
    scols = ["format_key", "variant", "role", "num_weights", "theoretical_bits", "theoretical_bits_per_weight", "payload_bytes",
             "padding_bits", "scale_bytes", "codec_metadata_bytes", "total_bytes", "actual_bits_per_weight",
             "compression_ratio_weights", "compression_ratio_payload_only", "non_quantized_param_bytes_fp32", "buffer_bytes",
             "full_model_bytes", "compression_ratio_full_model"]
    write_csv(out / "storage_summary.csv", storage_rows, scols)
    acols = ["format_key", "variant", "role", "val_accuracy_percent", "val_correct", "val_total", "val_cross_entropy",
             "val_drop_pp", "val_delta_ce"]
    write_csv(out / "clean_accuracy.csv", acc_rows, acols)

    frozen = hashlib.sha256(json.dumps({"checkpoint_sha256": ck_sha,
                                        "formats": {k: [v["config"], v["content_sha256"]] for k, v in fmt_manifest.items()}},
                                       sort_keys=True).encode()).hexdigest()
    manifest = {
        "stage": "J-FAMP runbook Stage 2 (candidate quantization and clean evaluation)",
        "checkpoint": {"path": str(ck.resolve()), "sha256": ck_sha, "size_bytes": ck.stat().st_size},
        "scope": "weight-only: Conv2d/Linear weights (21 tensors, %d weights) are quantized; biases, BatchNorm parameters, "
                 "running statistics and activations stay FP32" % n_total_w,
        "formats_excluded": {"INT3": "optional and deferred (runbook 4.1)"},
        "independence": "each format is generated from the original FP32 checkpoint, never from another format",
        "metric_definitions": {
            "sqnr_db": "10*log10(sum_i w_i^2 / sum_i (w_i - w^_i)^2), signal = FP32 weights, noise = FP32 weights minus decoded "
                       "weights; per layer, and GLOBAL pooled over all 21 tensors (sums of powers, paper Eq. 6); float64",
            "mse/rmse/mae/max_abs_error": "of w - w^ over the tensor (GLOBAL: pooled; MAE is the weight-weighted mean)",
            "relative_l2": "||w - w^||_2 / ||w||_2",
            "n_clipped": "weights whose magnitude exceeded the format's range before rounding (saturated)",
            "zeroed_from_nonzero": "weights that are nonzero in FP32 but decode to exactly 0",
            "accuracy": "top-1 correct / total; drop_pp = FP32 accuracy - candidate accuracy on the SAME images",
            "cross_entropy": "mean over images of the softmax cross-entropy, same images and criterion for all formats",
            "storage": "payload_bytes = actual bytes of the stored words; scale_bytes = float32 scales; padding_bits = bits in "
                       "payload beyond n*bits (INT4 odd count); total_bytes = payload + scales + codec metadata (none); "
                       "theoretical_bits = n * bits, reported separately",
            "compression_ratio_weights": "FP32 Conv/Linear weight bytes (%d) / total_bytes (payload + scales); excludes "
                                         "biases, BatchNorm parameters and buffers" % fp32_weight_bytes,
            "compression_ratio_full_model": "FP32 model bytes incl. non-quantized parameters and buffers (%d) / (total_bytes + "
                                            "non-quantized parameters %d B + buffers %d B, both left FP32)" % (fp32_full_bytes, nonq_param_bytes, buffer_bytes)},
        "evaluation": {"precision": "FP32 compute on decoded weights, model.eval(), inference_mode",
                       "validation": f"all {int(yv.numel())} held-out validation images (split seed {split_seed})",
                       "test": "official 10,000 test images, evaluated only by --stage test after this manifest exists",
                       "preprocessing": {"normalization": fdata.normalization_to_config(norm)}},
        "codec_source": source_info(), "codec_selftests": tests,
        "formats": fmt_manifest, "frozen_config_sha256": frozen,
        "environment": env_info(), "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    dump_json(manifest, manifest_path)
    print(f"\nfrozen configuration sha256: {frozen}")
    print("written:", out)


if __name__ == "__main__":
    main()
