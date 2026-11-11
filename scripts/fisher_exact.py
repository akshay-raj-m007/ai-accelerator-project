"""Exact-expectation diagonal Fisher on HELD-OUT VALIDATION images (primary Q of Stage 5).

Why: the training-image sampled-label Fisher is tiny for this network (saturated on its training images; the sampled-label estimator also undercounts, see
phase5/Q_ANALYSIS.md). Images: the FIRST ``--n`` images of the held-out validation split (default 2,048 = validation indices 0 .. 2,047). The Stage 5 end-to-end
clean evaluation then uses the remaining validation images (2,048 .. 4,999) so Q and the clean check never share an image. Never the test set.

  python scripts/fisher_exact.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch
Writes results/<run>/sensitivity_heldout/fisher_exact_val.pt, fisher_exact_val_layers.json, fisher_exact_val_meta.json
"""
import argparse
import pathlib
import time

import _common  # noqa: F401
from fr import data as fdata
from fr import sensitivity
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.mixed_run import split_params
from fr.model import load_model
from fr.utils import dump_json, env_info, git_info, resolve_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=2048, help="number of validation images (indices 0 .. n-1)")
    ap.add_argument("--min-prob", type=float, default=1e-8)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    out = pathlib.Path(args.out_dir) if args.out_dir else cfg.results_dir() / "sensitivity_heldout"
    out.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model(args.checkpoint)
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    loader = fdata.get_val_loader(cfg.data_path(), norm, 256, val_size, split_seed, cfg.num_workers, False, subset=args.n)
    t0 = time.time()
    res = sensitivity.compute_fisher_exact(model, loader, device, min_prob=args.min_prob)
    res.update({"source": "val", "val_index_range": [0, int(res["n_samples"])], "checkpoint_sha256": sha256_file(args.checkpoint),
                "val_size": val_size, "split_seed": split_seed})
    sensitivity.save_fisher(res, out / "fisher_exact_val.pt")
    dump_json({"checkpoint": str(args.checkpoint), "kind": "exact", "source": "val", "n_samples": res["n_samples"], "val_index_range": res["val_index_range"],
               "layers": sensitivity.layer_summary(res["fisher"])}, out / "fisher_exact_val_layers.json")
    meta = {k: v for k, v in res.items() if k != "fisher"}
    meta.update({"seconds": time.time() - t0, "env": env_info(), "git": git_info(), "normalization": norm})
    dump_json(meta, out / "fisher_exact_val_meta.json")
    tr = sum(float(f.double().sum()) for f in res["fisher"].values())
    print(f"exact Fisher on {res['n_samples']} validation images (indices 0..{res['n_samples'] - 1}): trace {tr:.4e}, {res['n_backward']} backward passes, "
          f"{time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
