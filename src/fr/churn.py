"""Prediction churn: a more sensitive clean metric than accuracy.

Top-1 accuracy nets out changes in both directions (an image that flips from right to wrong is cancelled by one
that flips from wrong to right), so tiny weight perturbations are invisible in it. Against a reference model
(the all-E4M11 baseline) we therefore also count

* ``flips``  images whose predicted class changed,
* ``c2w``    right -> wrong, ``w2c`` wrong -> right (accuracy change = w2c - c2w),
* ``kl``     mean KL( softmax(ref) || softmax(cfg) ) in nats (output-distribution shift),
* ``mean_abs_dlogit``  mean absolute logit difference.

All are deterministic functions of the quantized weights and the evaluation images.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

CHURN_FIELDS = ["config_id", "arm", "target_avg_bits", "method", "param", "acc", "flips", "c2w", "w2c",
                "kl", "mean_abs_dlogit"]


@torch.inference_mode()
def logits_tensors(model, X: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).float() for i in range(0, X.shape[0], batch_size)])


def churn_metrics(ref_logits: torch.Tensor, logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    rp, cp = ref_logits.argmax(1), logits.argmax(1)
    r_ok, c_ok = rp == y, cp == y
    lp_ref = torch.log_softmax(ref_logits.double(), 1)
    lp_cur = torch.log_softmax(logits.double(), 1)
    kl = (lp_ref.exp() * (lp_ref - lp_cur)).sum(1).mean().item()
    return {"acc": 100.0 * c_ok.float().mean().item(),
            "flips": int((rp != cp).sum()), "c2w": int((r_ok & ~c_ok).sum()),
            "w2c": int((~r_ok & c_ok).sum()), "kl": kl,
            "mean_abs_dlogit": (ref_logits - logits).abs().mean().item()}


def _append(path: Path, row: dict) -> None:
    new = not Path(path).exists()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CHURN_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def read_rows(path: Path) -> List[dict]:
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def render_report(meta: dict, val_rows: List[dict], churn: Dict[str, List[dict]], best: Dict[tuple, str]) -> str:
    """Markdown tables per split. ``churn[split]`` are rows of the churn CSV; ``val_rows`` carry SQNR/bits."""
    from .stability import spearman
    lines = ["# Prediction churn vs the all-E4M11 model", "",
             "* flips = images whose predicted class differs from the all-E4M11 model; c2w / w2c = right->wrong / "
             "wrong->right; KL = mean KL(softmax(E4M11) || softmax(config)) in nats.",
             "* Lower is closer to the E4M11 model. `random` rows: mean +/- std over the draws. The sets and "
             "budgets are the ones from the accuracy report.", ""]
    sq = {r["config_id"]: r for r in val_rows}
    for split, rows in churn.items():
        if not rows:
            continue
        by = {r["config_id"]: r for r in rows}
        lines += [f"## {split}", ""]
        lines += ["| reference | flips | c2w | w2c | KL | mean abs dlogit |", "|---|---|---|---|---|---|"]
        for cid in ["baseline|fp32"] + [f"baseline|{a}_all" for a in meta["arms"]]:
            if cid in by:
                r = by[cid]
                lines.append(f"| {cid.split('|')[1]} | {r['flips']} | {r['c2w']} | {r['w2c']} | "
                             f"{float(r['kl']):.2e} | {float(r['mean_abs_dlogit']):.4f} |")
        wins = {"lambda0": [0, 0], "product": [0, 0]}
        for arm in meta["arms"]:
            for target in meta["targets"]:
                key = (arm, float(target))
                lines += ["", f"### {arm}, target <= {target:g} bits/weight", "",
                          "| method | param | layers8 | flips | c2w | w2c | KL | mean abs dlogit |",
                          "|---|---|---|---|---|---|---|---|"]

                def row(label, cid):
                    if cid not in by:
                        return None
                    r = by[cid]
                    lines.append(f"| {label} | {r['param']} | {sq[cid]['n_layers8']} | {r['flips']} | {r['c2w']} | "
                                 f"{r['w2c']} | {float(r['kl']):.2e} | {float(r['mean_abs_dlogit']):.4f} |")
                    return r
                if key in best:
                    row("lambda (best on val acc)", best[key])
                r0 = row("Fisher only (lambda=0)", f"{arm}|T{target:g}|lambda|0")
                row("error only (lambda=1)", f"{arm}|T{target:g}|lambda|1")
                rp = row("product", f"{arm}|T{target:g}|product|")
                row("worst", f"{arm}|T{target:g}|worst|")
                rr = [by[c] for c in by if c.startswith(f"{arm}|T{target:g}|random|")]
                if rr:
                    fl = np.array([float(r["flips"]) for r in rr])
                    kl = np.array([float(r["kl"]) for r in rr])
                    lines.append(f"| random (n={len(rr)}) | - | - | {fl.mean():.0f} +/- {fl.std():.0f} | - | - | "
                                 f"{kl.mean():.2e} +/- {kl.std():.1e} | - |")
                    for name, r in (("lambda0", r0), ("product", rp)):
                        if r is not None:
                            wins[name][0] += int(float(r["kl"]) < kl.mean())
                            wins[name][1] += 1
        lines += ["", f"### Summary ({split})", ""]
        for name, (w, n) in wins.items():
            if n:
                lines.append(f"* {name}: KL below the random-draw mean in {w} of {n} (arm, target) cells")
        ids = [c for c in by if c in sq and sq[c].get("n_weights8") not in (None, "") and "baseline" not in c]
        if len(ids) > 5:
            kls = [float(by[c]["kl"]) for c in ids]
            lines += [f"* Spearman(KL, weights switched to 8 bit) = {spearman(kls, [float(sq[c]['n_weights8']) for c in ids]):.2f}; "
                      f"Spearman(KL, -SQNR) = {spearman(kls, [-float(sq[c]['sqnr_db']) for c in ids]):.2f} "
                      f"(over {len(ids)} configurations)"]
        lines.append("")
    return "\n".join(lines)
