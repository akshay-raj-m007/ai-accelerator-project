#!/usr/bin/env python
"""Stage 2, Experiment B (prediction-level impact) from the FROZEN encoded weights of stage2_quantize.py.

For FP32 and every format in the quantizer manifest, computes logits/predictions on the 5,000 validation images and records
changed predictions, agreement, FP32-correct->quantized-incorrect and incorrect->correct transitions, net change in correct
predictions, mean and maximum absolute logit difference. The encoded files are verified against the manifest content hashes first
(nothing is re-quantized, so Tables 7.1/7.2 are unaffected). Validation images only; the test set is never read.

  python scripts/stage2_prediction_impact.py --checkpoint $CK --run-name mobilenetv2_cifar10 \
      --out-dir reports/mobilenetv2_cifar10/stage_02_prediction_impact

Writes fp32_reference_predictions.npz (logits, predictions, labels, validation_indices), global_quantization_predictions.npz
(<format>_logits / <format>_predictions), prediction_impact_analysis.csv, fp32_reference_manifest.json.
"""
import argparse
import copy
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from fr import data as fdata
from fr import stage2 as S2
from fr.baseline_profile import sha256_file, sha256_ints
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, resolve_device

FIELDS = ["format_key", "variant", "role", "n", "changed_predictions", "agreement_percent", "correct_to_incorrect",
          "incorrect_to_correct", "net_change_correct", "mean_abs_logit_diff", "max_abs_logit_diff", "accuracy_percent", "correct"]


def compute_impact(logits_fp: np.ndarray, logits_q: np.ndarray, labels: np.ndarray) -> dict:
    p0, p1 = logits_fp.argmax(1), logits_q.argmax(1)
    c0, c1 = p0 == labels, p1 == labels
    d = np.abs(logits_q.astype(np.float64) - logits_fp.astype(np.float64))
    return {"n": int(labels.size), "changed_predictions": int((p0 != p1).sum()), "agreement_percent": 100.0 * float((p0 == p1).mean()),
            "correct_to_incorrect": int((c0 & ~c1).sum()), "incorrect_to_correct": int((~c0 & c1).sum()),
            "net_change_correct": int(c1.sum() - c0.sum()), "mean_abs_logit_diff": float(d.mean()), "max_abs_logit_diff": float(d.max()),
            "accuracy_percent": 100.0 * float(c1.mean()), "correct": int(c1.sum())}


@torch.inference_mode()
def logits_of(model, X, device, bs=500):
    model = model.to(device).eval()
    return torch.cat([model(X[i:i + bs]).float().cpu() for i in range(0, X.shape[0], bs)]).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None, help="default results/<run>/jfamp/quantization")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    qdir = Path(args.quant_dir) if args.quant_dir else cfg.results_dir() / "jfamp" / "quantization"
    out = Path(args.out_dir) if args.out_dir else cfg.results_dir() / "jfamp" / "prediction_impact"
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(args.checkpoint)
    man = json.load(open(qdir / "quantizer_manifest.json"))
    ck_sha = sha256_file(ck)
    if man["checkpoint"]["sha256"] != ck_sha:
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")

    model, ckpt = load_model(ck)
    model.eval()
    names = [f"{n}.weight" for n, _ in quantizable_modules(model)]
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=None)
    X, y = load_tensors(loader, device)
    labels = y.cpu().numpy()
    va_idx = np.asarray(fdata.split_indices(50000, val_size, split_seed)[1], dtype=np.int64)

    lg0 = logits_of(model, X, device)
    np.savez_compressed(out / "fp32_reference_predictions.npz", logits=lg0, predictions=lg0.argmax(1), labels=labels, validation_indices=va_idx)
    dump_json({"checkpoint_sha256": ck_sha, "validation_images": int(labels.size), "validation_indices_sha256": sha256_ints(va_idx),
               "fp32_correct": int((lg0.argmax(1) == labels).sum()), "fp32_accuracy_percent": 100.0 * float((lg0.argmax(1) == labels).mean()),
               "precision": "FP32, model.eval(), inference_mode", "quantizer_frozen_config_sha256": man["frozen_config_sha256"]},
              out / "fp32_reference_manifest.json")

    reg = S2.codec_registry()
    rows, arrays = [], {}
    for key, fm in man["formats"].items():
        enc = S2.load_encoded(qdir / fm["encoded_file"], names)
        if S2.content_sha256(enc) != fm["content_sha256"]:
            sys.exit(f"{key}: encoded weights differ from the frozen content hash")
        m = copy.deepcopy(model).eval()
        params = dict(m.named_parameters())
        with torch.no_grad():
            for n, e in enc.items():
                params[n].copy_(torch.from_numpy(np.ascontiguousarray(reg[key].decode(e))).to(params[n].dtype))
        lg = logits_of(m, X, device)
        arrays[f"{key}_logits"], arrays[f"{key}_predictions"] = lg, lg.argmax(1)
        r = {"format_key": key, "variant": fm["config"]["variant"], "role": fm["role"], **compute_impact(lg0, lg, labels)}
        rows.append(r)
        print(f"{key:11s} changed {r['changed_predictions']:5d}  agreement {r['agreement_percent']:6.2f}%  C->I {r['correct_to_incorrect']:4d}  "
              f"I->C {r['incorrect_to_correct']:4d}  mean|dlogit| {r['mean_abs_logit_diff']:.5f}  max {r['max_abs_logit_diff']:.4f}")
    np.savez_compressed(out / "global_quantization_predictions.npz", **arrays)
    with open(out / "prediction_impact_analysis.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("written:", out)


if __name__ == "__main__":
    main()
