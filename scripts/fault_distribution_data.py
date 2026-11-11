"""Collect the DATA for the clean-vs-faulted weight/activation distribution comparison of two formats (GPU machine; plots are made locally).

For each format the worst RECORDED Stage 4 trial of one shared configuration (same layer, fault group, k) is replayed exactly (same encoded
elements and bits from fault_trials.jsonl). The script measures, for the clean and the faulted model:
  * the weights of the faulted layer, the output activations of the faulted layer and the logits, as histograms on FIXED bin edges
    (the same edges for every format, so the formats can be compared on one axis);
  * the largest finite |output| of every quantizable layer, and non-finite counts;
  * accuracy, and whether the replay reproduces the accuracy recorded in Stage 4.
Output: fault_distribution_data.npz (small, a few hundred KB) in --out-dir. Plot it with scripts/plot_stage4_summary.py.

Default: formats FP16 and E4M11-SN, group exponent, k = 8, layer chosen as the one with the largest sum of the two formats' mean accuracy drops.

  python scripts/fault_distribution_data.py --checkpoint $FR_ROOT/checkpoints/scratch/best.pth --run-name scratch \
      --screening-dir reports/scratch/jfamp/fault_screening --out-dir reports/scratch/jfamp/fault_screening
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
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules
from fr.utils import resolve_device

LIN, HI = 0.05, 1.0e5


def fixed_edges(lin=LIN, hi=HI, n_lin=80, per_decade=24):
    """Bin edges shared by all formats: linear inside [-lin, lin], logarithmic outside up to +-hi (use a symlog axis)."""
    n_log = int(round(np.log10(hi / lin) * per_decade))
    pos = np.logspace(np.log10(lin), np.log10(hi), n_log)
    return np.concatenate([-pos[::-1], np.linspace(-lin, lin, n_lin)[1:-1], pos])


@torch.inference_mode()
def collect(model, X, target, edges, batch_size=250):
    model.eval()
    mods = quantizable_modules(model)
    maxabs, nonfin = [0.0] * len(mods), [0] * len(mods)
    act_counts = np.zeros(len(edges) - 1, dtype=np.int64)
    act_stat = {"n": 0, "s1": 0.0, "s2": 0.0, "max": 0.0}
    logits = []
    hooks = []

    def mk(i):
        def hook(mod, inp, out):
            t = out.detach().float().reshape(-1)
            fin = torch.isfinite(t)
            nonfin[i] += int((~fin).sum())
            tf = t[fin]
            if tf.numel():
                maxabs[i] = max(maxabs[i], float(tf.abs().max()))
            if i == target:
                v = tf.double()
                act_stat["n"] += int(v.numel())
                act_stat["s1"] += float(v.sum())
                act_stat["s2"] += float((v * v).sum())
                act_stat["max"] = max(act_stat["max"], float(v.abs().max()))
                act_counts[:] += np.histogram(tf.cpu().numpy(), bins=edges)[0]
        return hook

    for i, (_, m) in enumerate(mods):
        hooks.append(m.register_forward_hook(mk(i)))
    try:
        for s in range(0, X.shape[0], batch_size):
            logits.append(model(X[s:s + batch_size]).detach().float().cpu())
    finally:
        for h in hooks:
            h.remove()
    lg = torch.cat(logits).reshape(-1)
    fin = torch.isfinite(lg)
    lv = lg[fin].numpy()
    n = act_stat["n"]
    mean = act_stat["s1"] / n
    return {"maxabs": np.array(maxabs), "nonfinite": np.array(nonfin), "act_counts": act_counts,
            "act_n": n, "act_mean": mean, "act_std": float(np.sqrt(max(act_stat["s2"] / n - mean ** 2, 0))), "act_max": act_stat["max"],
            "logit_counts": np.histogram(lv, bins=edges)[0], "logit_n": int(lv.size), "logit_mean": float(lv.mean()), "logit_std": float(lv.std()),
            "logit_max": float(np.abs(lv).max()), "logits_nonfinite": int((~fin).sum())}


def pick_layer(rows, fmts, group, k):
    d = {}
    for r in rows:
        if r["scope_kind"] == "weights" and r["fault_group"] == group and int(r["k"]) == k and r["format"] in fmts and r["layer_name"] != "GLOBAL":
            d.setdefault(r["layer_name"], {})[r["format"]] = float(r["drop_pp_mean"])
    cand = {n: sum(v.values()) for n, v in d.items() if len(v) == len(fmts)}
    return max(cand, key=cand.get)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--formats", nargs=2, default=["fp16", "e4m11_sn"])
    ap.add_argument("--group", default="exponent")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--layer", default=None)
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--screening-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-images", type=int, default=1000)
    args = ap.parse_args()
    cfg = config_from_args(args)
    device = resolve_device(cfg.device)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    sdir = Path(args.screening_dir) if args.screening_dir else base / "fault_screening"
    out = Path(args.out_dir) if args.out_dir else sdir
    out.mkdir(parents=True, exist_ok=True)

    man2 = json.load(open(qdir / "quantizer_manifest.json"))
    ck = Path(args.checkpoint)
    if man2["checkpoint"]["sha256"] != sha256_file(ck):
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    rows = list(csv.DictReader(open(sdir / "fault_summary.csv")))
    recs = [json.loads(l) for l in open(sdir / "fault_trials.jsonl")]
    model, ckpt = load_model(ck)
    model.eval()
    mods = quantizable_modules(model)
    names = [n for n, _ in mods]
    layer_names = [f"{n}.weight" for n in names]
    lname = args.layer or pick_layer(rows, args.formats, args.group, args.k)
    li = names.index(lname)
    print(f"layer {lname} (id {li}), group {args.group}, k = {args.k}, formats {args.formats}")

    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    loader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed, cfg.num_workers, False, subset=args.n_images)
    X, y = load_tensors(loader, device)
    edges = fixed_edges()

    save = {"edges": edges, "layer_names": np.array(names), "faulted_layer": np.array(lname), "group": np.array(args.group), "k": np.array(args.k),
            "n_images": np.array(int(y.numel())), "formats": np.array(args.formats)}
    for fmt in args.formats:
        cand = [r for r in recs if r["format"] == fmt and r["scope_kind"] == "weights" and r["layer_name"] == lname
                and r["fault_group"] == args.group and r["requested_k"] == args.k]
        if not cand:
            sys.exit(f"no recorded trials for {fmt}/{lname}/{args.group}/k{args.k}")
        rec = max(cand, key=lambda r: r["clean_accuracy"] - r["fault_accuracy"])
        stored = S2.load_encoded(qdir / man2["formats"][fmt]["encoded_file"], layer_names)
        if S2.content_sha256(stored) != man2["formats"][fmt]["content_sha256"]:
            sys.exit(f"{fmt}: encoded weights differ from the frozen content hash")
        inj = S3.EncodedFaultInjector(model, S3.EncodedStore(fmt, stored), S2.make_codecs(True)[fmt], device)
        clean_eval = S3.evaluate(inj.model, X, y)
        clean = collect(inj.model, X, li, edges)
        w_clean = inj.flat[li].detach().cpu().numpy().astype(np.float64).copy()
        flips = [(f[1], f[2], f[3]) for f in rec["flips"] if f[0] == "w"]
        elems = sorted({e for _, e, _ in flips})
        before = [float(inj.flat[li][e]) for e in elems]
        inj.inject(weight_flips=flips)
        try:
            after = [float(inj.flat[li][e]) for e in elems]
            w_fault = inj.flat[li].detach().cpu().numpy().astype(np.float64).copy()
            fault_eval = S3.evaluate(inj.model, X, y)
            fault = collect(inj.model, X, li, edges)
        finally:
            inj.restore()
        ok = None
        if int(y.numel()) == rec["sample_count"]:
            ok = abs(fault_eval["accuracy"] - rec["fault_accuracy"]) < 1e-6
            print(f"{fmt}: replayed {rec['trial_id']}: fault accuracy {fault_eval['accuracy']:.2f} vs recorded {rec['fault_accuracy']:.2f} -> "
                  f"{'REPRODUCED' if ok else 'MISMATCH'}")
        p = fmt + "__"
        save[p + "trial_id"] = np.array(rec["trial_id"])
        save[p + "clean_accuracy"] = np.array(clean_eval["accuracy"])
        save[p + "fault_accuracy"] = np.array(fault_eval["accuracy"])
        save[p + "replay_reproduces_record"] = np.array(-1 if ok is None else int(ok))
        save[p + "flipped_elements"] = np.array(elems)
        save[p + "flipped_old"] = np.array(before)
        save[p + "flipped_new"] = np.array(after)
        for tag, w in (("clean", w_clean), ("fault", w_fault)):
            save[p + f"w_{tag}_counts"] = np.histogram(w, bins=edges)[0]
            save[p + f"w_{tag}_stats"] = np.array([w.size, w.mean(), w.std(), np.abs(w).max()])
        for tag, c in (("clean", clean), ("fault", fault)):
            save[p + f"a_{tag}_counts"] = c["act_counts"]
            save[p + f"a_{tag}_stats"] = np.array([c["act_n"], c["act_mean"], c["act_std"], c["act_max"], int(c["nonfinite"][li])])
            save[p + f"l_{tag}_counts"] = c["logit_counts"]
            save[p + f"l_{tag}_stats"] = np.array([c["logit_n"], c["logit_mean"], c["logit_std"], c["logit_max"], c["logits_nonfinite"]])
            save[p + f"maxabs_{tag}"] = c["maxabs"]
            save[p + f"nonfinite_{tag}"] = c["nonfinite"]
        del inj
        if device.type == "cuda":
            torch.cuda.empty_cache()
    path = out / "fault_distribution_data.npz"
    np.savez_compressed(path, **save)
    print("written", path, f"({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
