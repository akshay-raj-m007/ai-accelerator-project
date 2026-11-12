"""Measure the protection option ``scale_tmr`` (scale word stored 3 x, bit-wise majority vote on read; see src/fr/protect.py). GPU machine, about 10 minutes.

Protocol, mirrors the Stage 4 layer-wise scale screening so the protected value is comparable with the unprotected R_scale of the candidate table:
  for each INT format (INT8, INT4), each of the 21 layers, 20 independent trials: ONE random bit flip in the layer's stored scale (bit uniform over the 3 x 32 stored bits
  of the protected store); accuracy on the first 1,000 validation images. A PAIRED CONTROL applies the equivalent single flip (copy 0, same bit) to the unprotected store
  in the same trial, so the effect of the protection is visible on identical fault draws.
Also (no images): an exhaustive check on the stored bytes of every layer - all 96 single flips, and all two-flip combinations in two different copies - recording how many
are corrected (single flips: all; two flips: all except the same bit position in two copies, the documented limit).

  python scripts/stage5_tmr_layerwise.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
Writes results/<run>/jfamp/optimization/tmr_layerwise_summary.csv, tmr_layerwise_trials.jsonl, tmr_exhaustive_check.json
"""
import argparse
import csv
import itertools
import json
import sys
from collections import OrderedDict
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from fr import data as fdata
from fr import protect as P
from fr import stage2 as S2
from fr import stage3 as S3
from fr import stage5_e2e as E2
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, env_info, git_info, resolve_device

INT_FORMATS = ["int8_pt", "int4_pt"]


def exhaustive_check(store: E2.MixedStore):
    """Single flips and cross-copy double flips on the stored bytes: how many leave the voted scale unchanged."""
    rows = []
    for l, cp in enumerate(store.scale_copies):
        orig = int(cp[0])
        single_ok = single_n = double_ok = double_n = same_bit_wrong = 0
        for c in range(3):
            for b in range(32):
                t = cp.copy()
                t[c] ^= np.uint32(1 << b)
                single_n += 1
                single_ok += int(P.vote_u32(*t) == orig)
        for (c1, c2) in itertools.combinations(range(3), 2):
            for b1 in range(32):
                for b2 in range(32):
                    t = cp.copy()
                    t[c1] ^= np.uint32(1 << b1)
                    t[c2] ^= np.uint32(1 << b2)
                    double_n += 1
                    ok = int(P.vote_u32(*t) == orig)
                    double_ok += ok
                    same_bit_wrong += int(b1 == b2 and not ok)
        rows.append({"layer": store.names[l], "single_flips_tested": single_n, "single_flips_corrected": single_ok, "double_flips_tested": double_n,
                     "double_flips_corrected": double_ok, "double_flips_uncorrected_same_bit_position": same_bit_wrong})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--out-dir", default=None, help="default results/<run>/jfamp/optimization")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--n-images", type=int, default=1000)
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    out = Path(args.out_dir) if args.out_dir else base / "optimization"
    out.mkdir(parents=True, exist_ok=True)
    man2 = json.load(open(qdir / "quantizer_manifest.json"))
    ck = Path(args.checkpoint)
    ck_sha = sha256_file(ck)
    if man2["checkpoint"]["sha256"] != ck_sha:
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    model, ckpt = load_model(ck)
    model.eval()
    mods = quantizable_modules(model)
    lnames = [f"{n}.weight" for n, _ in mods]
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    X, y = load_tensors(fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=args.n_images), device)
    codecs = S2.make_codecs(True)
    stores = {}
    for f in INT_FORMATS:
        enc = S2.load_encoded(qdir / man2["formats"][f]["encoded_file"], lnames)
        if S2.content_sha256(enc) != man2["formats"][f]["content_sha256"]:
            sys.exit(f"{f}: encoded weights differ from the frozen content hash")
        stores[f] = S3.EncodedStore(f, enc)
    ctx = {"experiment_id": "jfamp-resnet18-cifar10-stage5-scale-tmr", "run_id": "tmr-layerwise", "model": "resnet18_cifar", "dataset": f"CIFAR-10 validation, first {int(y.numel())} images",
           "checkpoint_hash": ck_sha, "codec_version": f"scale_tmr|{S3.INJECTOR_VERSION}|{E2.SCHEMA_VERSION}"}

    # ---- exhaustive check on the stored bytes (no images)
    chk = {}
    for f in INT_FORMATS:
        chk[f] = exhaustive_check(E2.MixedStore([f + "+scale_tmr"] * 21, stores))
    tot = {f: {k: sum(r[k] for r in rows) for k in rows[0] if k != "layer"} for f, rows in chk.items()}
    dump_json({"note": "stored-byte check, no model: single flips in any copy/bit are all corrected; two flips in two different copies are corrected unless they hit the SAME bit position",
               "totals": tot, "per_layer": chk}, out / "tmr_exhaustive_check.json")
    print("exhaustive check:", json.dumps(tot))

    # ---- layer-wise measured trials, protected vs paired unprotected control
    trials_path = out / "tmr_layerwise_trials.jsonl"
    rows_out, f_log = [], open(trials_path, "w", encoding="utf8")
    for f in INT_FORMATS:
        prot = E2.MixedStore([f + "+scale_tmr"] * 21, stores)
        plain = E2.MixedStore([f] * 21, stores)
        inj_p = E2.MixedFaultInjector(model, prot, codecs, device)
        inj_u = E2.MixedFaultInjector(model, plain, codecs, device)
        clean = S3.evaluate(inj_p.model, X, y)
        assert abs(clean["accuracy"] - S3.evaluate(inj_u.model, X, y)["accuracy"]) < 1e-9          # same clean model
        for l, name in enumerate(lnames):
            dp, du = [], []
            for t in range(args.trials):
                rng = np.random.default_rng(S3.trial_seed(cfg.seed, f"tmr:{f}", name, "scales", "all", 1, t))
                c, b = int(rng.integers(0, 3)), int(rng.integers(0, 32))
                rp = E2.run_mixed_trial(inj_p, X, y, clean, ctx, assignment_id=f"{f}+scale_tmr|{name}", dist="scales_k1", k=1, trial=t, base_seed=cfg.seed, explicit_scale_flips=[(l, c, b)])
                ru = E2.run_mixed_trial(inj_u, X, y, clean, ctx, assignment_id=f"{f}|{name}", dist="scales_k1", k=1, trial=t, base_seed=cfg.seed, explicit_scale_flips=[(l, 0, b)])
                for r in (rp, ru):
                    if not r["restored_exactly"] or not r["flip_applied_to_encoded_bytes"]:
                        sys.exit(f"{f}/{name}/t{t}: restoration or flip check failed")
                    f_log.write(json.dumps(r) + "\n")
                dp.append(clean["accuracy"] - rp["fault_accuracy"])
                du.append(clean["accuracy"] - ru["fault_accuracy"])
            dp, du = np.array(dp), np.array(du)
            rows_out.append({"format": f, "layer_id": l, "layer_name": name[:-len(".weight")], "n_trials": args.trials, "sample_count": int(y.numel()), "clean_accuracy": clean["accuracy"],
                             "protected_drop_pp_mean": float(dp.mean()), "protected_drop_pp_std": float(dp.std(ddof=1)), "protected_drop_pp_p90": float(np.percentile(dp, 90)),
                             "protected_drop_pp_worst": float(dp.max()), "unprotected_control_drop_pp_mean": float(du.mean()), "unprotected_control_drop_pp_p90": float(np.percentile(du, 90)),
                             "unprotected_control_drop_pp_worst": float(du.max())})
        print(f"{f}: 21 layers done; protected mean over layers {np.mean([r['protected_drop_pp_mean'] for r in rows_out if r['format'] == f]):.3f} pp, "
              f"unprotected control {np.mean([r['unprotected_control_drop_pp_mean'] for r in rows_out if r['format'] == f]):.2f} pp")
    f_log.close()
    with open(out / "tmr_layerwise_summary.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    dump_json({"checkpoint_sha256": ck_sha, "trials": args.trials, "images": int(y.numel()), "base_seed": cfg.seed, "protection": "scale_tmr: 3 stored copies, bit-wise majority on read",
               "overhead_bytes_per_layer": P.TMR_OVERHEAD_BYTES, "git": git_info(), "environment": env_info()}, out / "tmr_manifest.json")
    print("written to", out)


if __name__ == "__main__":
    main()
