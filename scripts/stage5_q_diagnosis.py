"""Why is the Fisher-weighted quantization penalty Q so small? A decomposition (diagnostic, not part of the runbook outputs).

For a weight perturbation d = w_hat - w (all Conv/Linear layers of one datatype) the clean cross-entropy change on a validation set is, with g = grad L, H = Hessian of L:

    dL(d)  =  g.d  +  1/2 d^T H d  +  O(|d|^3)               (what really happens, exact for t -> 0)
    Q-based prediction = 1/2 sum_i F_ii d_i^2                (F = diagonal true Fisher; the Fisher replaces H and its diagonal replaces the full matrix)

Measured here on the same validation images (no test data): the actual dL(t d) for t = 0.25, 0.5, 1; g.d; d^T H d (exact Hessian-vector product);
1/2 d^T diag(F_val) d with the Fisher estimated on validation images; 1/2 d^T diag(F_train) d with the Fisher on TRAINING images (what the candidate table uses);
and the confidence statistics that explain the size of F (mean max-probability, mean (1 - p_max)^2) on both image sets.

  python scripts/stage5_q_diagnosis.py --checkpoint models/scratch/best.pth --formats int4_pt int8_pt fp8_e4m3fn --device cpu
"""
import argparse
import json
import math
from collections import OrderedDict

import _common  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as Fn
from fr import data as fdata
from fr import sensitivity
from fr import stage2 as S2
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, resolve_device


def confidence(model, X, y, device):
    with torch.no_grad():
        p = torch.softmax(model(X.to(device)).float(), 1)
    pm, pred = p.max(1)
    ce = -torch.log(p.gather(1, y.to(device).view(-1, 1)).clamp_min(1e-30)).squeeze(1)
    return {"n": int(y.numel()), "accuracy": float((pred.cpu() == y.cpu()).float().mean()), "mean_ce": float(ce.mean()), "mean_max_prob": float(pm.mean()),
            "mean_one_minus_pmax_sq": float(((1 - pm) ** 2).mean()), "frac_pmax_gt_0.999": float((pm > 0.999).float().mean())}


def loss_at(model, params, base, dirs, t, X, y, bs=250):
    with torch.no_grad():
        for p, b, d in zip(params, base, dirs):
            p.copy_(b + t * d)
    tot = 0.0
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            tot += float(Fn.cross_entropy(model(X[i:i + bs]), y[i:i + bs], reduction="sum"))
    with torch.no_grad():
        for p, b in zip(params, base):
            p.copy_(b)
    return tot / X.shape[0]


def grad_and_hvp(model, params, dirs, X, y, bs=100):
    """g = grad of mean CE; Hd = Hessian-vector product with the direction d (exact, double backward)."""
    N = X.shape[0]
    g_tot = [torch.zeros_like(p) for p in params]
    Hd_tot = [torch.zeros_like(p) for p in params]
    for i in range(0, N, bs):
        loss = Fn.cross_entropy(model(X[i:i + bs]), y[i:i + bs], reduction="sum") / N
        g = torch.autograd.grad(loss, params, create_graph=True)
        gd = sum((gi * di).sum() for gi, di in zip(g, dirs))
        Hd = torch.autograd.grad(gd, params)
        for a, b in zip(g_tot, g):
            a += b.detach()
        for a, b in zip(Hd_tot, Hd):
            a += b.detach()
    return g_tot, Hd_tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--formats", nargs="+", default=["int4_pt", "int8_pt", "fp8_e4m3fn"])
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-fisher", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    model, ckpt = load_model(args.checkpoint)
    model.to(device).eval()
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    Xv, yv = load_tensors(fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=args.n_val), device)
    Xt, yt = load_tensors(fdata.get_calibration_loader(cfg.data_path(), norm, args.n_fisher, val_size, split_seed, batch_size=args.n_fisher), device)
    out = {"n_val": int(yv.numel()), "n_fisher": args.n_fisher, "confidence": {"train_calibration_images": confidence(model, Xt, yt, device),
                                                                              "validation_images": confidence(model, Xv, yv, device)}}
    print("confidence:", json.dumps(out["confidence"], indent=1))

    # true Fisher (sampled labels) on training images and on validation images
    def fisher_on(X, y):
        ld = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X.cpu(), y.cpu()), batch_size=64)
        return sensitivity.compute_fisher(model, ld, device, kind="true", seed=0, max_samples=args.n_fisher)["fisher"]

    Ft = fisher_on(Xt, yt)
    Fv = fisher_on(Xv[:args.n_fisher], yv[:args.n_fisher])
    names = list(Ft)
    tr_t = {k: float(v.double().sum()) for k, v in Ft.items()}
    tr_v = {k: float(v.double().sum()) for k, v in Fv.items()}
    out["fisher_trace"] = {"train_images": sum(tr_t.values()), "validation_images": sum(tr_v.values()), "ratio_val_over_train": sum(tr_v.values()) / sum(tr_t.values())}
    print("Fisher trace: train %.4e  validation %.4e  ratio %.1f" % (out["fisher_trace"]["train_images"], out["fisher_trace"]["validation_images"], out["fisher_trace"]["ratio_val_over_train"]))

    mods = quantizable_modules(model)
    params = [m.weight for _, m in mods]
    for p in params:
        p.requires_grad_(True)
    base = [p.detach().clone() for p in params]
    codecs = S2.make_codecs(True)
    out["formats"] = {}
    for fmt in args.formats:
        dirs = []
        for (n, m), b in zip(mods, base):
            w = b.cpu().numpy()
            what = codecs[fmt].decode(codecs[fmt].encode(w)).reshape(w.shape)
            dirs.append(torch.from_numpy(np.ascontiguousarray(what)).to(device) - b)
        g, Hd = grad_and_hvp(model, params, dirs, Xv, yv)
        gd = float(sum((a * b).sum().double() for a, b in zip(g, dirs)))
        dHd = float(sum((a * b).sum().double() for a, b in zip(Hd, dirs)))
        dFd_v = float(sum((Fv[k].to(device).double() * d.double() ** 2).sum() for k, d in zip(names, dirs)))
        dFd_t = float(sum((Ft[k].to(device).double() * d.double() ** 2).sum() for k, d in zip(names, dirs)))
        d_norm2 = float(sum((d.double() ** 2).sum() for d in dirs))
        L0 = loss_at(model, params, base, dirs, 0.0, Xv, yv)
        act = {t: loss_at(model, params, base, dirs, t, Xv, yv) - L0 for t in (0.25, 0.5, 1.0)}
        row = {"actual_dL_t0.25": act[0.25], "actual_dL_t0.5": act[0.5], "actual_dL_t1": act[1.0], "first_order_g_dot_d": gd, "half_dHd_full_hessian": 0.5 * dHd,
               "taylor_2nd_order_g.d+1/2dHd": gd + 0.5 * dHd, "half_dFd_diag_fisher_VAL": 0.5 * dFd_v, "half_dFd_diag_fisher_TRAIN_candidate_table": 0.5 * dFd_t,
               "d_norm2": d_norm2, "relative_perturbation_|d|/|w|": math.sqrt(d_norm2 / float(sum((b.double() ** 2).sum() for b in base)))}
        out["formats"][fmt] = row
        print(f"\n{fmt}: |d|/|w| = {row['relative_perturbation_|d|/|w|']:.3f}")
        for k, v in row.items():
            if k not in ("d_norm2",):
                print(f"  {k:46s} {v:+.4e}")
    dump_json(out, args.out or "q_diagnosis.json")


if __name__ == "__main__":
    main()
