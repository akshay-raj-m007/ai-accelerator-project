"""Feature maps of one validation image through ResNet-18 stages for FP32 and the weights stored in E4M11-SN, FP8 E4M3FN and INT4 (illustration, not a runbook output).

All Conv/Linear weights of a model are replaced by the decoded frozen Stage 2 encoding of that format (activations stay FP32), exactly as in the uniform single-format
models of Stage 2. Hooks record the output of the stem (conv1+BN+ReLU) and of layer1 .. layer4.

  python scripts/plot_feature_maps.py --checkpoint models/scratch/best.pth --out plots_preview/stage5 --index 0
Writes 12_feature_maps_one_image.png and 13_feature_map_error_vs_depth.png.
"""
import argparse
import copy
from pathlib import Path

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import torch
from fr import data as fdata
from fr import stage2 as S2
from fr.config import add_config_args, config_from_args
from fr.faults import load_tensors
from fr.mixed_run import split_params
from fr.model import load_model, quantizable_modules

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
FORMATS = [("fp32", "FP32 (reference)", "#52514e"), ("e4m11_sn", "E4M11-SN", "#2a78d6"), ("fp8_e4m3fn", "FP8 E4M3FN", "#1baf7a"), ("int4_pt", "INT4", "#e34948")]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STAGES = [("stem", "stem\n(conv1)", 64, 32), ("layer1", "layer1", 64, 32), ("layer2", "layer2", 128, 16), ("layer3", "layer3", 256, 8), ("layer4", "layer4", 512, 4)]
plt.rcParams.update({"figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK, "font.family": "DejaVu Sans"})


def quantized_copy(model, fmt, codecs):
    m = copy.deepcopy(model).eval()
    if fmt == "fp32":
        return m
    with torch.no_grad():
        for _, mod in quantizable_modules(m):
            w = mod.weight.detach().numpy()
            mod.weight.copy_(torch.from_numpy(np.ascontiguousarray(codecs[fmt].decode(codecs[fmt].encode(w)).reshape(w.shape))))
    return m


def features(model, x):
    out = {}
    hooks = [model.relu.register_forward_hook(lambda m, i, o: out.__setitem__("stem", o.detach()))]
    for n in ("layer1", "layer2", "layer3", "layer4"):
        hooks.append(getattr(model, n).register_forward_hook(lambda m, i, o, n=n: out.__setitem__(n, o.detach())))
    with torch.no_grad():
        logits = model(x)
    for h in hooks:
        h.remove()
    out["logits"] = logits
    return out


TAG = ""


def P(name):
    """File name with the model tag as prefix, so figures of several models can live side by side."""
    return f"{TAG}_{name}" if TAG else name


def main():
    global TAG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="plots_preview", help="root folder; figures go to <out>/<model-tag>/stage5/<model-tag>_<name>.png")
    ap.add_argument("--model-tag", default="resnet18_cifar10", help="model + dataset tag used in folder and file names")
    ap.add_argument("--index", type=int, default=None, help="validation image index; default: first image (of the first 300) correctly classified by all four models")
    ap.add_argument("--only-fig12", action="store_true", help="draw only the feature-map figure (fast)")
    ap.add_argument("--n-error", type=int, default=500, help="validation images for the error-versus-depth figure")
    ap.add_argument("--opt-dir", default="reports/scratch/jfamp/optimization_round1", help="folder with q_layerwise_validation.csv (measured per-layer loss change)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    TAG = args.model_tag
    out = Path(args.out) / args.model_tag / "stage5"
    out.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model(args.checkpoint)
    model.eval()
    norm = fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))
    val_size, split_seed = split_params(ckpt, cfg)
    X, y = load_tensors(fdata.get_val_loader(cfg.data_path(), norm, 250, val_size, split_seed, 0, False, subset=max(args.n_error, 300)), "cpu")
    codecs = S2.make_codecs(True)
    models = {k: quantized_copy(model, k, codecs) for k, _, _ in FORMATS}
    with torch.no_grad():
        preds = {k: torch.cat([m(X[i:i + 250]).argmax(1) for i in range(0, len(X), 250)]) for k, m in models.items()}
    if args.index is None:
        ok = np.ones(300, dtype=bool)
        for k in models:
            ok &= (preds[k][:300] == y[:300]).numpy()
        idx = int(np.argmax(ok))
    else:
        idx = args.index
    x1, label = X[idx:idx + 1], int(y[idx])
    feats = {k: features(m, x1) for k, m in models.items()}
    mean, std = (torch.tensor(norm[0]).view(1, 3, 1, 1), torch.tensor(norm[1]).view(1, 3, 1, 1)) if norm else (0, 1)
    img = (x1 * std + mean)[0].permute(1, 2, 0).clamp(0, 1).numpy()
    print(f"image {idx} class {CLASSES[label]}; predictions:", {k: CLASSES[int(preds[k][idx])] for k in preds})

    # ------------------------------------------------------------ figure 12: feature maps, 3 channels per stage
    ncol_ch = 3
    nf = len(FORMATS)
    ratios = []
    for g in range(ncol_ch):
        ratios += [1] * nf + ([0.45] if g < ncol_ch - 1 else [])
    fig = plt.figure(figsize=(15.5, 11.6))
    gs = fig.add_gridspec(len(STAGES), len(ratios), width_ratios=ratios, hspace=0.14, wspace=0.05, top=0.72, bottom=0.03, left=0.115, right=0.995)
    fig.text(0.012, 0.985, f"What do the early and late layers see?  Feature maps of one validation image ({CLASSES[label]}), weights stored in different formats", fontsize=14.5, fontweight="bold", va="top")
    fig.text(0.012, 0.95, "Each row is one stage of ResNet-18 (spatial size shrinks, channel count grows). Each of the three column groups is one channel per stage (the 1st, 2nd and 3rd strongest in FP32, so the channel\n"
             "numbers differ between rows); inside a group the 4 squares are that same channel for FP32, E4M11-SN, FP8 and INT4 weights, on one colour scale (bright = strong response). Only the weights are quantized;\n"
             "activations stay FP32. Early stages look like the photo (edges, colour blobs); later stages are coarse and abstract. The size of the differences is quantified in figures 13 and 14.", fontsize=9.6, color=INK2, va="top", linespacing=1.4)
    ax0 = fig.add_axes([0.015, 0.765, 0.085, 0.085])
    ax0.imshow(img, interpolation="nearest")
    ax0.set_title(f"input: {CLASSES[label]}", fontsize=9)
    ax0.axis("off")
    row0 = {}
    for r, (key, name, nch, hw) in enumerate(STAGES):
        f32 = feats["fp32"][key][0]
        top = torch.argsort(f32.pow(2).sum((1, 2)), descending=True)[:ncol_ch].tolist()
        for ci, ch in enumerate(top):
            vmax = float(f32[ch].max()) or 1.0
            for fi, (fk, fname, col) in enumerate(FORMATS):
                ax = fig.add_subplot(gs[r, ci * (nf + 1) + fi])
                ax.imshow(feats[fk][key][0][ch].numpy(), cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color(col)
                    s.set_linewidth(1.6)
                if r == 0:
                    ax.set_title(fname.replace(" (reference)", "").replace("FP8 E4M3FN", "FP8"), fontsize=7.6, color=INK2)
                    row0.setdefault(ci, []).append(ax)
            ax_last = fig.axes[-1]
            ax_last.set_xlabel(f"ch {ch}", fontsize=7.4, color=INK2, loc="left") if False else None
        fig.text(0.012, 0.72 - (r + 0.5) * (0.69 / len(STAGES)), f"{name}\n{nch} ch, {hw}x{hw}", fontsize=9.2, fontweight="bold", va="center", ha="left")
    for ci, axs_ in row0.items():
        x0, x1 = axs_[0].get_position().x0, axs_[-1].get_position().x1
        ytop = axs_[0].get_position().y1 + 0.028
        fig.text((x0 + x1) / 2, ytop + 0.004, f"channel group {ci + 1}  (same channel, 4 formats)", ha="center", va="bottom", fontsize=9.4, fontweight="bold")
        fig.add_artist(matplotlib.lines.Line2D([x0, x1], [ytop, ytop], color=INK2, lw=1.2))
    fig.savefig(out / P("12_feature_maps_one_image.png"), dpi=150)
    plt.close(fig)
    print("wrote", out / P("12_feature_maps_one_image.png"))
    if args.only_fig12:
        return

    # ------------------------------------------------------------ figure 13: how much do the features change (many images)
    Xe = X[:args.n_error]
    ref = {}
    err = {k: {s[0]: [] for s in STAGES} | {"logits": []} for k, _, _ in FORMATS if k != "fp32"}
    for i in range(0, len(Xe), 100):
        xb = Xe[i:i + 100]
        fr = features(models["fp32"], xb)
        for k in err:
            fq = features(models[k], xb)
            for s in list(err[k]):
                a, b = fr[s].flatten(1), fq[s].flatten(1)
                err[k][s].append((((b - a).norm(dim=1)) / a.norm(dim=1).clamp_min(1e-12)).numpy())
    names = [s[0] for s in STAGES] + ["logits"]
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.075, right=0.985)
    fig.text(0.012, 0.975, "How far do the feature maps move when the weights are quantized?", fontsize=14.5, fontweight="bold", va="top")
    fig.text(0.012, 0.925, f"Relative change of the feature map, ||f_quantized - f_FP32|| / ||f_FP32||, mean over {len(Xe)} validation images (all layers in the same format). Each stage shows the accumulated effect of the\n"
             "weight error of all layers up to that stage.\nE4M11-SN is practically lossless (about 1e-4); FP8 changes the features by 4-10 %; INT4 by 18-41 %. The change rises through the stages (INT4 peaks at layer3) and is partly averaged out at the output.",
             fontsize=9.6, color=INK2, va="top", linespacing=1.4)
    w = 0.26
    for j, (k, fname, col) in enumerate([f for f in FORMATS if f[0] != "fp32"]):
        vals = [float(np.concatenate(err[k][n]).mean()) for n in names]
        xs = np.arange(len(names)) + (j - 1) * w
        ax.bar(xs, vals, width=w - 0.02, color=col, label=fname.replace("FP8 E4M3FN", "FP8 E4M3FN"))
        for xv, v in zip(xs, vals):
            ax.text(xv, v * 1.12, f"{v:.1e}" if v < 0.01 else f"{v:.3f}", ha="center", fontsize=7.8, color=INK2, rotation=90 if v < 0.01 else 0, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["stem", "layer1", "layer2", "layer3", "layer4", "logits\n(output)"])
    ax.set_ylabel("relative feature change (log scale)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.savefig(out / P("13_feature_map_error_vs_depth.png"), dpi=150)
    plt.close(fig)
    print("wrote", out / P("13_feature_map_error_vs_depth.png"))
    for k in err:
        print(k, {n: round(float(np.concatenate(err[k][n]).mean()), 5) for n in names})

    # ------------------------------------------------------------ figure 14: only ONE layer quantized (INT4): change of the network output
    mods = quantizable_modules(model)
    with torch.no_grad():
        base_logits = torch.cat([model(Xe[i:i + 100]) for i in range(0, len(Xe), 100)])
    rel, nw = [], []
    for n, m in mods:
        orig = m.weight.detach().clone()
        w = orig.numpy()
        with torch.no_grad():
            m.weight.copy_(torch.from_numpy(np.ascontiguousarray(codecs["int4_pt"].decode(codecs["int4_pt"].encode(w)).reshape(w.shape))))
            lg = torch.cat([model(Xe[i:i + 100]) for i in range(0, len(Xe), 100)])
            m.weight.copy_(orig)
        rel.append(float(((lg - base_logits).norm(dim=1) / base_logits.norm(dim=1)).mean()))
        nw.append(int(orig.numel()))
    labels = [n.replace("layer", "L").replace(".downsample.0", ".ds").replace("conv", "c") for n, _ in mods]
    import csv as _csv
    qv = Path(args.opt_dir) / "q_layerwise_validation.csv"
    dce = {r["layer_name"]: float(r["measured_dCE"]) for r in _csv.DictReader(open(qv, encoding="utf8")) if r["datatype"] == "int4_pt"} if qv.exists() else {}
    fig, (a, c, b) = plt.subplots(1, 3, figsize=(16, 8), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.8], "wspace": 0.08, "top": 0.8, "bottom": 0.07, "left": 0.07, "right": 0.985})
    fig.text(0.012, 0.975, "Quantize ONE layer to INT4: how far does the output move, and does the loss go up?", fontsize=14.5, fontweight="bold", va="top")
    fig.text(0.012, 0.925, f"Left: relative change of the 10 output logits, mean over {len(Xe)} validation images (only the named layer is INT4, all others FP32). Middle: change of the validation cross-entropy loss, same\n"
             "setting (images 2,048-4,999). Right: number of weights. The big late layers move the output the MOST, but in 3 of the 4 biggest movers the loss goes DOWN (logits become less over-confident);\n"
             "the small early layers move the output less but raise the loss. Q measures the loss effect, which is why it ranks early layers as more sensitive.", fontsize=9.6, color=INK2, va="top", linespacing=1.4)
    yy = np.arange(len(mods))
    a.barh(yy, rel, color="#e34948", height=0.62)
    a.set_xscale("log")
    for yi, v in zip(yy, rel):
        a.text(v * 1.08, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    a.set_xlabel("relative change of the output logits (log scale)")
    a.set_title("output movement", fontsize=10.5, loc="left", fontweight="bold")
    vals = np.array([dce.get(n, 0.0) for n, _ in mods])
    c.barh(yy[vals >= 0], vals[vals >= 0], color="#6250d6", height=0.62)
    c.barh(yy[vals < 0], vals[vals < 0], color="#b9b8b2", height=0.62)
    c.axvline(0, color=INK2, lw=0.8)
    for yi, v in zip(yy, vals):
        c.text(v + (0.0005 if v >= 0 else -0.0005), yi, f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.8, color=INK2)
    c.set_xlim(-0.026, 0.016)
    c.set_xlabel("change of validation loss (grey: loss went down)")
    c.set_title("loss change (what Q measures)", fontsize=10.5, loc="left", fontweight="bold")
    b.barh(yy, np.array(nw) / 1e6, color="#2a78d6", height=0.62)
    b.set_xlabel("weights in the layer (millions)")
    b.set_title("size of the layer", fontsize=10.5, loc="left", fontweight="bold")
    a.set_yticks(yy)
    a.set_yticklabels(labels, fontsize=9)
    a.invert_yaxis()
    for ax_ in (a, b, c):
        ax_.grid(axis="x", color=GRID, lw=0.8)
        ax_.set_axisbelow(True)
        for s in ("top", "right"):
            ax_.spines[s].set_visible(False)
        ax_.tick_params(axis="y", length=0)
    fig.savefig(out / P("14_one_layer_int4_output_change.png"), dpi=150)
    plt.close(fig)
    print("wrote", out / P("14_one_layer_int4_output_change.png"))
    print({l: round(r, 4) for l, r in zip(labels, rel)})


if __name__ == "__main__":
    main()
