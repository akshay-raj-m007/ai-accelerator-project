"""Layer-wise validation of the Stage 5 quantization penalty Q (GPU machine, a few minutes).

Q(l,d) = sum F (w - w_hat)^2 predicts that storing ONLY layer l in datatype d raises the clean cross-entropy by 0.5 Q(l,d). This script measures it: for every layer l and
datatype d, only layer l is replaced by its decoded frozen Stage 2 encoding (all other layers stay FP32) and the model is evaluated on the validation images 2,048..4,999
(2,952 images, DISJOINT from the Fisher images 0..2047 behind the primary Q). Both Q columns are compared: Q_fisher_val_exact (primary) and Q_fisher_train_sampled
(alternative). The all-layers change of each datatype is measured here on the same images, which gives an ADDITIVITY check (sum over layers of the single-layer changes
versus the all-layers change). Values below the noise of an evaluation on a few thousand images (a few 1e-4 in cross-entropy) carry no ranking information; the
numbers are reported as measured.

  python scripts/stage5_q_validation.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from fr import data as fdata
from fr import stage2 as S2
from fr import stage3 as S3
from fr import stage5 as S5
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.layer_scores import spearman
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, resolve_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--opt-dir", default=None, help="folder with jfamp_candidate_table.csv (default results/<run>/jfamp/optimization)")
    ap.add_argument("--images", type=int, default=None, help="validation images after the skip (default: all remaining; dry runs only)")
    ap.add_argument("--clean-skip", type=int, default=2048, help="skip the first N validation images (the Fisher images)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    odir = Path(args.opt_dir) if args.opt_dir else base / "optimization"
    man2 = json.load(open(qdir / "quantizer_manifest.json"))
    ck = Path(args.checkpoint)
    if man2["checkpoint"]["sha256"] != sha256_file(ck):
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    table = list(csv.DictReader(open(odir / "jfamp_candidate_table.csv")))
    Q = {(r["datatype"], r["layer_name"]): float(r["Q_fisher_val_exact"]) for r in table}
    Qt = {(r["datatype"], r["layer_name"]): float(r["Q_fisher_train_sampled"]) for r in table}

    model, ckpt = load_model(ck)
    model.to(device).eval()
    mods = quantizable_modules(model)
    names = [n for n, _ in mods]
    lnames = [f"{n}.weight" for n in names]
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=args.images, skip=args.clean_skip)
    X, y = load_tensors(loader, device)
    fp32 = S3.evaluate(model, X, y)
    print(f"FP32 on {int(y.numel())} validation images (skipping the first {args.clean_skip}): {fp32['accuracy']:.2f} % / CE {fp32['loss']:.5f}")

    codecs = S2.make_codecs(True)
    rows, all_layers = [], {}
    for fmt in S5.FORMATS:
        stored = S2.load_encoded(qdir / man2["formats"][fmt]["encoded_file"], lnames)
        if S2.content_sha256(stored) != man2["formats"][fmt]["content_sha256"]:
            sys.exit(f"{fmt}: encoded weights differ from the frozen content hash")
        decoded = [torch.from_numpy(np.ascontiguousarray(codecs[fmt].decode(stored[ln]).reshape(m.weight.shape))).to(device) for ln, (_, m) in zip(lnames, mods)]
        origs = [m.weight.detach().clone() for _, m in mods]
        with torch.no_grad():                                           # all layers in this datatype
            for (_, m), d in zip(mods, decoded):
                m.weight.copy_(d)
        ev_all = S3.evaluate(model, X, y)
        with torch.no_grad():
            for (_, m), o in zip(mods, origs):
                m.weight.copy_(o)
        all_layers[fmt] = {"measured_dCE": ev_all["loss"] - fp32["loss"], "measured_acc_drop_pp": fp32["accuracy"] - ev_all["accuracy"]}
        for l, (n, m) in enumerate(mods):                                # one layer at a time
            with torch.no_grad():
                m.weight.copy_(decoded[l])
            try:
                ev = S3.evaluate(model, X, y)
            finally:
                with torch.no_grad():
                    m.weight.copy_(origs[l])
            rows.append({"datatype": fmt, "layer_id": l, "layer_name": n, "Q_fisher_val_exact": Q[(fmt, n)], "predicted_dCE": 0.5 * Q[(fmt, n)],
                         "Q_fisher_train_sampled": Qt[(fmt, n)], "predicted_dCE_train": 0.5 * Qt[(fmt, n)], "measured_dCE": ev["loss"] - fp32["loss"],
                         "measured_acc_drop_pp": fp32["accuracy"] - ev["accuracy"], "val_samples": int(y.numel())})
        print(f"{fmt}: 21 single-layer evaluations and the all-layers evaluation done")

    out = {"note": "one layer quantized at a time, others FP32; validation images 2,048..4,999 only (disjoint from the Fisher images); primary Q = exact-expectation validation Fisher",
           "val_samples": int(y.numel()), "val_skip": args.clean_skip, "fp32": {"accuracy": fp32["accuracy"], "loss": fp32["loss"]}, "per_datatype": {}}
    for fmt in S5.FORMATS:
        rs = [r for r in rows if r["datatype"] == fmt]
        pred = np.array([r["predicted_dCE"] for r in rs])
        predt = np.array([r["predicted_dCE_train"] for r in rs])
        meas = np.array([r["measured_dCE"] for r in rs])
        pos = meas > 0
        ratio = meas[pos] / pred[pos] if pos.any() else np.array([])
        am = all_layers[fmt]["measured_dCE"]
        out["per_datatype"][fmt] = {
            "spearman_Q_vs_measured_dCE_over_21_layers": spearman(pred, meas), "spearman_Qtrain_vs_measured_dCE_over_21_layers": spearman(predt, meas),
            "sum_predicted_dCE": float(pred.sum()), "sum_predicted_dCE_train_fisher": float(predt.sum()), "sum_measured_dCE_single_layer": float(meas.sum()),
            "all_layers_measured_dCE": am, "all_layers_measured_acc_drop_pp": all_layers[fmt]["measured_acc_drop_pp"],
            "sum_predicted_over_all_layers_measured": float(pred.sum() / am) if am > 0 else None,
            "n_layers_with_positive_measured_dCE": int(pos.sum()), "median_measured_over_predicted": float(np.median(ratio)) if ratio.size else None,
            "max_abs_measured_dCE": float(np.abs(meas).max())}
    out["spearman_pooled_105_candidates"] = spearman([r["predicted_dCE"] for r in rows], [r["measured_dCE"] for r in rows])
    out["spearman_pooled_105_candidates_train_fisher"] = spearman([r["predicted_dCE_train"] for r in rows], [r["measured_dCE"] for r in rows])
    dump_json(out, odir / "q_layerwise_validation.json")
    with open(odir / "q_layerwise_validation.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nformat       Spearman over 21 layers (val Q / train Q)   sum 1/2 Q (val / train)    sum measured (single layers)   all layers measured")
    for fmt, v in out["per_datatype"].items():
        print(f"{fmt:11s}  {v['spearman_Q_vs_measured_dCE_over_21_layers']:+.3f} / {v['spearman_Qtrain_vs_measured_dCE_over_21_layers']:+.3f}"
              f"                      {v['sum_predicted_dCE']:.2e} / {v['sum_predicted_dCE_train_fisher']:.2e}      {v['sum_measured_dCE_single_layer']:+.2e}"
              f"                   {v['all_layers_measured_dCE']:+.2e}")
    print(f"pooled Spearman over 105 candidates: val Q {out['spearman_pooled_105_candidates']:+.3f}, train Q {out['spearman_pooled_105_candidates_train_fisher']:+.3f} -> {odir / 'q_layerwise_validation.json'}")


if __name__ == "__main__":
    main()
