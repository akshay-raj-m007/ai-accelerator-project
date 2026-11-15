#!/usr/bin/env python
"""MobileNetV2 Stage 6, Fisher quantization penalty (CPU, about a minute; needs no images and no GPU).

Q_fisher_val_exact(l, d) = sum_i F_i (w_i - w_hat_i)^2   (the ResNet-18 optimiser's PRIMARY Q, src/fr/stage5.py::fisher_q)
  F      = diagonal Fisher of the FP32 model, expectation over labels computed EXACTLY, on held-out VALIDATION images 0..2047 (scripts/fisher_exact.py)
  w_hat  = the decoded frozen Stage 2 encoding of layer l in format d (verified against the content hash of the quantizer manifest)
0.5 * Q is the second-order prediction of the clean cross-entropy increase when only layer l is stored in format d.

  python scripts/stage6_fisher_q.py --checkpoint $CK --run-name mobilenetv2_cifar10 \
      --fisher $FR_ROOT/results/mobilenetv2_cifar10/sensitivity_heldout/fisher_exact_val.pt --out-dir reports/mobilenetv2_cifar10/stage_06_fisher

Writes fisher_q.csv (53 layers x 7 formats), fisher_q_meta.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from fr import stage2 as S2
from fr import stage5 as S5
from fr import stage6 as S6
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.model import load_model, quantizable_modules
from fr.sensitivity import load_fisher
from fr.utils import dump_json, git_info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fisher", default=None, help="default results/<run>/sensitivity_heldout/fisher_exact_val.pt")
    ap.add_argument("--quant-dir", default=None, help="default results/<run>/jfamp/quantization")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-images", type=int, default=2048, help="refuse a Fisher computed on fewer validation images")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    base = cfg.results_dir()
    fpath = Path(args.fisher) if args.fisher else base / "sensitivity_heldout" / "fisher_exact_val.pt"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "jfamp" / "quantization"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(args.checkpoint)
    ck_sha = sha256_file(ck)
    man = json.load(open(qdir / "quantizer_manifest.json"))
    if man["checkpoint"]["sha256"] != ck_sha:
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    fv = load_fisher(fpath)
    if fv.get("kind") != "exact" or fv.get("source") != "val":
        sys.exit(f"{fpath}: the primary Q needs the exact-expectation Fisher on validation images (scripts/fisher_exact.py), got kind={fv.get('kind')!r} source={fv.get('source')!r}")
    if int(fv.get("n_samples", 0)) < args.min_images:
        sys.exit(f"{fpath}: Fisher computed on {fv.get('n_samples')} images, at least {args.min_images} required")
    if fv.get("checkpoint_sha256") not in (None, ck_sha):
        sys.exit("the Fisher was computed on a different checkpoint")
    F = fv["fisher"]
    model, _ = load_model(ck)
    mods = quantizable_modules(model)
    names = [n for n, _ in mods]
    lnames = [f"{n}.weight" for n in names]
    if list(F) != lnames:
        sys.exit("Fisher layer names / order differ from the model's Conv2d/Linear weights")
    W = {f"{n}.weight": m.weight.detach().cpu().numpy() for n, m in mods}
    codecs = S2.make_codecs(True)
    rows = []
    for fmt in S6.FORMATS:
        stored = S2.load_encoded(qdir / man["formats"][fmt]["encoded_file"], lnames)
        if S2.content_sha256(stored) != man["formats"][fmt]["content_sha256"]:
            sys.exit(f"{fmt}: encoded weights differ from the frozen content hash")
        for lid, (n, ln) in enumerate(zip(names, lnames)):
            what = codecs[fmt].decode(stored[ln]).reshape(W[ln].shape)
            q = S5.fisher_q(W[ln], what, F[ln].numpy())
            rows.append({"layer_index": lid, "layer_name": n, "format_key": fmt, "num_weights": int(W[ln].size), "Q_fisher_val_exact": q, "Q_pred_dCE_fisher": 0.5 * q,
                         "fisher_trace": float(F[ln].double().sum())})
        print(f"{fmt}: sum Q {sum(r['Q_fisher_val_exact'] for r in rows if r['format_key'] == fmt):.4e}  (predicted all-layer dCE {0.5 * sum(r['Q_fisher_val_exact'] for r in rows if r['format_key'] == fmt):.4e})")
    with open(out / "fisher_q.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    dump_json({"Q_definition": S5.Q_DEFINITION, "checkpoint_sha256": ck_sha, "stage2_frozen_config_sha256": man["frozen_config_sha256"], "fisher_file": str(fpath),
               "fisher_sha256": sha256_file(fpath), "fisher_kind": fv.get("kind"), "fisher_source": fv.get("source"), "fisher_n_samples": fv.get("n_samples"),
               "fisher_val_index_range": fv.get("val_index_range"), "fisher_min_prob": fv.get("min_prob"), "rows": len(rows), "source_commit": git_info()}, out / "fisher_q_meta.json")
    print(f"{len(rows)} rows -> {out / 'fisher_q.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
