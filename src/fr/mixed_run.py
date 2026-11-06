"""Pipelines for the mixed-precision experiments (Fisher -> layer scores -> selection -> clean metrics).

Phases (all resumable, CSV rows are appended as they finish):

``run_fisher``  diagonal Fisher on a calibration subset of the *training* split.
``run_sweep``   for every (8-bit format, size target, scorer) choose the layers, quantize, evaluate on the
                *validation* split. The test set is not touched. Also records all-E4M11 / all-8-bit references.
``run_final``   re-evaluates the saved selections that matter (lambda chosen on validation, the two lambda
                end points, product, worst, every random draw) on the *test* split and writes the report.

Selections are saved to ``selections.json`` so the test phase (and the later fault step) use exactly the layer
sets that were chosen during the validation phase.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import allocation, data as fdata, faults, formats8, mixed, sensitivity
from .config import Config
from .model import load_model
from .quantize import conv_linear_weights
from .sqnr import sqnr_db
from .utils import dump_json, env_info, resolve_device

log = logging.getLogger("fr.mixed_run")

SWEEP_FIELDS = ["config_id", "arm", "target_avg_bits", "method", "param", "avg_bits_payload",
                "avg_bits_with_metadata", "n_layers8", "n_weights8", "acc", "ce", "sqnr_db", "tol_used",
                "n_feasible", "total_cost", "layers8"]
TEST_FIELDS = SWEEP_FIELDS + ["val_acc", "val_ce", "is_lambda_best"]
DEFAULT_ARMS = ("int8_pt", "int8_pc", "fp8", "fp8_scaled")
DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def split_params(ckpt: dict, cfg: Config):
    """(val_size, split_seed) of the training run, from ``checkpoint['config']`` when it records them."""
    c = ckpt.get("config") or {}
    if "val_size" not in c or "seed" not in c:
        log.warning("checkpoint config has no val_size/seed; using %d/%d. If this checkpoint was trained on "
                    "other splits, validation/calibration images may overlap its training data.",
                    cfg.val_size, cfg.seed)
    return int(c.get("val_size", cfg.val_size)), int(c.get("seed", cfg.seed))


def _norm_for(cfg: Config, ckpt: dict):
    return fdata.resolve_normalization(cfg.normalize, "checkpoint", ckpt.get("config"))


# ------------------------------------------------------------------------------------------- Fisher
def run_fisher(cfg: Config, checkpoint, kinds: Sequence[str] = ("true", "empirical"),
               calib_size: int = 1024, out_dir: Optional[Path] = None) -> dict:
    device = resolve_device(cfg.device)
    model, ckpt = load_model(checkpoint)
    norm = _norm_for(cfg, ckpt)
    val_size, split_seed = split_params(ckpt, cfg)
    out_dir = Path(out_dir or cfg.results_dir() / "sensitivity")
    loader = fdata.get_calibration_loader(cfg.data_path(), norm, calib_size, val_size, split_seed,
                                          batch_size=64, num_workers=cfg.num_workers)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoint": str(checkpoint), "calib_size": calib_size, "val_size": val_size,
               "split_seed": split_seed, "fisher_seed": cfg.seed, "device": str(device), "env": env_info(),
               "normalization": fdata.normalization_to_config(norm), "kinds": {}}
    for kind in kinds:
        res = sensitivity.compute_fisher(model, loader, device, kind=kind, seed=cfg.seed)
        sensitivity.save_fisher(res, out_dir / f"fisher_{kind}.pt")
        layers = sensitivity.layer_summary(res["fisher"])
        summary["kinds"][kind] = {"n_samples": res["n_samples"], "mean_nll_on_calibration": res["mean_nll"],
                                  "layers": layers}
        dump_json({"checkpoint": str(checkpoint), "kind": kind, "n_samples": res["n_samples"],
                   "layers": layers}, out_dir / f"fisher_{kind}_layers.json")
        log.info("Fisher[%s]: %d samples, mean NLL %.4f", kind, res["n_samples"], res["mean_nll"])
        for name, s in layers.items():
            log.info("  %-28s n=%9d  trace=%.4e  trace/param=%.4e", name, s["count"], s["trace"],
                     s["trace_per_param"])
    dump_json(summary, out_dir / "sensitivity_meta.json")
    return summary


# ------------------------------------------------------------------------------------ evaluation
def pooled_sqnr(original: Dict[str, np.ndarray], deq: Dict[str, np.ndarray]) -> float:
    """Paper Eq. 6 pooled over all given tensors (float64)."""
    sig = err = 0.0
    for k, w in original.items():
        a = w.astype(np.float64).reshape(-1)
        d = a - deq[k].astype(np.float64).reshape(-1)
        sig += float(np.dot(a, a))
        err += float(np.dot(d, d))
    return sqnr_db(sig, err)


def evaluate_assignment(model, assignment: Dict[str, str], X, y, device, qcache: dict,
                        weights: Dict[str, np.ndarray], e4m11_mode: str, bias: int) -> dict:
    q, buf = mixed.quantize_mixed(model, assignment, e4m11_mode, bias, cache=qcache)
    m = faults.eval_tensors(q.to(device), X, y)
    acc = mixed.bit_accounting(buf)
    n8 = sum(v for k, v in acc["weights_by_format"].items() if k != mixed.BASE)
    return {"acc": m["accuracy_percent"], "ce": m["cross_entropy"],
            "sqnr_db": pooled_sqnr(weights, conv_linear_weights(q)),
            "avg_bits_payload": acc["avg_bits_payload"],
            "avg_bits_with_metadata": acc["avg_bits_with_metadata"],
            "n_weights8": n8, "n_layers8": sum(1 for b in buf.values() if b["format"] != mixed.BASE)}


def _read_rows(path: Path) -> List[dict]:
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _append(path: Path, fields: Sequence[str], row: dict) -> None:
    new = not Path(path).exists()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _fmt_param(x) -> str:
    return "" if x is None else (f"{x:g}" if isinstance(x, float) else str(x))


def config_id(arm: str, target: float, method: str, param) -> str:
    return f"{arm}|T{target:g}|{method}|{_fmt_param(param)}"


def _scorers(lambdas: Sequence[float], random_draws: int):
    out = [("lambda", float(l)) for l in lambdas]
    out += [("product", None), ("worst", None)]
    out += [("random", d) for d in range(random_draws)]
    return out


def _select(method, param, rows, names, sizes, target, norm_how, fisher_mode, tol):
    if method == "lambda":
        costs = allocation.combined_score(rows, param, norm_how, fisher_mode)
        return allocation.select("min_cost", names, sizes, target, costs=costs, tol=tol)
    if method == "product":
        return allocation.select("min_cost", names, sizes, target,
                                 costs=[r["damage"] for r in rows], tol=tol)
    if method == "worst":
        return allocation.select("worst", names, sizes, target,
                                 damages=[r["damage"] for r in rows], tol=tol)
    return allocation.select("random", names, sizes, target, tol=tol, seed=int(param))


# ------------------------------------------------------------------------------------------- sweep
def run_sweep(cfg: Config, checkpoint, fisher_path, arms: Sequence[str] = DEFAULT_ARMS,
              targets: Sequence[float] = (14.0, 12.0, 10.0), lambdas: Sequence[float] = DEFAULT_LAMBDAS,
              random_draws: int = 5, tol: float = 0.05, norm_how: str = "rank",
              fisher_mode: str = "per_param", e4m11_mode: str = "subnormal", bias: int = 7,
              val_subset: Optional[int] = None, out_dir: Optional[Path] = None) -> dict:
    device = resolve_device(cfg.device)
    model, ckpt = load_model(checkpoint)
    norm = _norm_for(cfg, ckpt)
    val_size, split_seed = split_params(ckpt, cfg)
    out_dir = Path(out_dir or cfg.results_dir() / "mixed")
    out_dir.mkdir(parents=True, exist_ok=True)

    vloader = fdata.get_val_loader(cfg.data_path(), norm, cfg.eval_batch_size, val_size, split_seed,
                                   cfg.num_workers, False, subset=val_subset)
    X, y = faults.load_tensors(vloader, device)
    weights = conv_linear_weights(model)
    fres = sensitivity.load_fisher(fisher_path)
    fisher = sensitivity.fisher_numpy(fres["fisher"])
    if list(fisher) != list(weights):
        raise ValueError("Fisher file does not match the model's Conv/Linear layers")
    names, sizes = list(weights), [w.size for w in weights.values()]

    csv_path, sel_path = out_dir / "mixed_val.csv", out_dir / "selections.json"
    done = {r["config_id"] for r in _read_rows(csv_path)}
    selections = json.load(open(sel_path)) if sel_path.exists() else {}
    meta = {"checkpoint": str(checkpoint), "fisher_file": str(fisher_path), "fisher_kind": fres["kind"],
            "fisher_samples": fres["n_samples"], "arms": list(arms), "targets": list(targets),
            "lambdas": list(lambdas), "random_draws": random_draws, "tol": tol, "norm": norm_how,
            "fisher_mode": fisher_mode, "e4m11_mode": e4m11_mode, "bias": bias,
            "val_images": int(X.shape[0]), "val_size": val_size, "split_seed": split_seed,
            "layers": names, "layer_sizes": sizes, "env": env_info()}
    dump_json(meta, out_dir / "sweep_meta.json")

    qcache: dict = {}
    eval_cache: dict = {}

    def run_one(cid, arm, target, method, param, assignment, sel):
        sig = (arm, tuple(sorted(assignment)))
        if sig not in eval_cache:
            eval_cache[sig] = evaluate_assignment(model, assignment, X, y, device, qcache, weights,
                                                  e4m11_mode, bias)
        r = eval_cache[sig]
        row = {"config_id": cid, "arm": arm, "target_avg_bits": target, "method": method,
               "param": _fmt_param(param), **r,
               "tol_used": sel.get("tol_used", ""), "n_feasible": sel.get("n_feasible", ""),
               "total_cost": "" if sel.get("total_cost") is None else sel["total_cost"],
               "layers8": ";".join(sorted(assignment))}
        _append(csv_path, SWEEP_FIELDS, row)
        done.add(cid)
        log.info("%-46s val %.2f%%  CE %.4f  SQNR %.2f dB  bits %.2f  layers8=%d", cid, r["acc"], r["ce"],
                 r["sqnr_db"], r["avg_bits_payload"], r["n_layers8"])

    # references
    fp = faults.eval_tensors(model.to(device), X, y)
    model.cpu()
    if "baseline|fp32" not in done:
        _append(csv_path, SWEEP_FIELDS, {"config_id": "baseline|fp32", "arm": "fp32", "method": "baseline",
                                         "acc": fp["accuracy_percent"], "ce": fp["cross_entropy"],
                                         "avg_bits_payload": 32, "avg_bits_with_metadata": 32})
        done.add("baseline|fp32")
    if "baseline|e4m11_all" not in done:
        run_one("baseline|e4m11_all", "e4m11", "", "baseline", "all", {}, {})
    for arm in arms:
        cid = f"baseline|{arm}_all"
        if cid not in done:
            run_one(cid, arm, "", "baseline", "all", {n: arm for n in names}, {})

    for arm in arms:
        rows = allocation.layer_table(weights, fisher, arm, cache={})
        for target in targets:
            for method, param in _scorers(lambdas, random_draws):
                cid = config_id(arm, target, method, param)
                sel = selections.get(cid)
                if sel is None:
                    sel = _select(method, param, rows, names, sizes, float(target), norm_how, fisher_mode, tol)
                    sel = {k: v for k, v in sel.items() if k != "mask"}
                    selections[cid] = sel
                    dump_json(selections, sel_path)
                if cid in done:
                    continue
                run_one(cid, arm, target, method, param, {n: arm for n in sel["layers8"]}, sel)
    dump_json({"layer_table": {arm: allocation.layer_table(weights, fisher, arm, cache={}) for arm in arms}},
              out_dir / "layer_table.json")
    return {"csv": str(csv_path), "selections": str(sel_path), "meta": meta}


# ------------------------------------------------------------------------------------------- final
def _f(x, d=2):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "-"


def pick_lambda_best(val_rows: List[dict]) -> Dict[tuple, str]:
    """(arm, target) -> config_id of the lambda with the best validation accuracy (ties: lower CE, then lower
    lambda, i.e. more weight on Fisher)."""
    best: Dict[tuple, tuple] = {}
    for r in val_rows:
        if r["method"] != "lambda":
            continue
        key = (r["arm"], float(r["target_avg_bits"]))
        score = (float(r["acc"]), -float(r["ce"]), -float(r["param"]))
        if key not in best or score > best[key][0]:
            best[key] = (score, r["config_id"])
    return {k: v[1] for k, v in best.items()}


def run_final(cfg: Config, checkpoint, sweep_dir: Optional[Path] = None,
              test_subset: Optional[int] = None) -> dict:
    device = resolve_device(cfg.device)
    sweep_dir = Path(sweep_dir or cfg.results_dir() / "mixed")
    val_rows = _read_rows(sweep_dir / "mixed_val.csv")
    if not val_rows:
        raise FileNotFoundError(f"run the validation sweep first: {sweep_dir / 'mixed_val.csv'} is missing")
    selections = json.load(open(sweep_dir / "selections.json"))
    sweep_meta = json.load(open(sweep_dir / "sweep_meta.json"))
    e4m11_mode, bias = sweep_meta["e4m11_mode"], sweep_meta["bias"]

    model, ckpt = load_model(checkpoint)
    norm = _norm_for(cfg, ckpt)
    loader = fdata.get_test_loader(cfg.data_path(), norm, cfg.eval_batch_size, cfg.num_workers,
                                   test_subset or cfg.test_subset, False)
    X, y = faults.load_tensors(loader, device)
    weights = conv_linear_weights(model)

    best = pick_lambda_best(val_rows)
    best_ids = set(best.values())
    by_id = {r["config_id"]: r for r in val_rows}
    wanted = []
    for r in val_rows:
        if r["method"] == "baseline":
            wanted.append(r["config_id"])
        elif r["method"] in ("product", "worst", "random"):
            wanted.append(r["config_id"])
        elif r["method"] == "lambda" and (r["config_id"] in best_ids or float(r["param"]) in (0.0, 1.0)):
            wanted.append(r["config_id"])

    csv_path = sweep_dir / "mixed_test.csv"
    done = {r["config_id"] for r in _read_rows(csv_path)}
    qcache: dict = {}
    for cid in wanted:
        if cid in done:
            continue
        v = by_id[cid]
        if cid == "baseline|fp32":
            m = faults.eval_tensors(model.to(device), X, y)
            model.cpu()
            r = {"acc": m["accuracy_percent"], "ce": m["cross_entropy"], "sqnr_db": "",
                 "avg_bits_payload": 32, "avg_bits_with_metadata": 32, "n_weights8": 0, "n_layers8": 0}
        else:
            if v["method"] == "baseline":
                assignment = {} if v["arm"] == "e4m11" else {n: v["arm"] for n in weights}
            else:
                assignment = {n: v["arm"] for n in selections[cid]["layers8"]}
            r = evaluate_assignment(model, assignment, X, y, device, qcache, weights, e4m11_mode, bias)
        row = dict(v)
        row.update(r)
        row.update({"val_acc": v["acc"], "val_ce": v["ce"], "is_lambda_best": int(cid in best_ids)})
        _append(csv_path, TEST_FIELDS, row)
        log.info("%-46s test %.2f%% (val %.2f%%)", cid, r["acc"], float(v["acc"]))
    test_rows = _read_rows(csv_path)
    md = render_report(val_rows, test_rows, sweep_meta, selections)
    (sweep_dir / "mixed_report.md").write_text(md, encoding="utf-8")
    return {"csv": str(csv_path), "report": str(sweep_dir / "mixed_report.md")}


def render_report(val_rows, test_rows, meta, selections) -> str:
    t = {r["config_id"]: r for r in test_rows}
    v = {r["config_id"]: r for r in val_rows}
    n_val = meta["val_images"]
    se = 100.0 * (0.95 * 0.05 / n_val) ** 0.5
    lines = ["# Mixed-precision (E4M11 + 8-bit) results", "",
             f"* Fisher: `{meta['fisher_kind']}`, {meta['fisher_samples']} calibration images (training split); "
             f"score normalization `{meta['norm']}`, Fisher `{meta['fisher_mode']}`; E4M11 baseline "
             f"`{meta['e4m11_mode']}` bias {meta['bias']}.",
             f"* Selection and lambda choice use the validation split ({n_val} images, standard error about "
             f"{se:.2f} pp at 95% accuracy): differences below ~{2 * se:.1f} pp are not significant.",
             "* Test accuracy is reported for the saved selections only; lambda was chosen on validation.",
             "* `random` rows: mean +/- std over draws; `worst` = feasible set with the largest total damage "
             "(sanity envelope). avg bits = payload bits per weight; +meta also counts scale metadata.", ""]
    fp = t.get("baseline|fp32") or v.get("baseline|fp32")
    base = t.get("baseline|e4m11_all")
    lines += ["## References", "", "| config | val acc % | test acc % | SQNR dB | avg bits |", "|---|---|---|---|---|"]
    for cid in ["baseline|fp32", "baseline|e4m11_all"] + [f"baseline|{a}_all" for a in meta["arms"]]:
        if cid in v:
            lines.append(f"| {cid.split('|')[1]} | {_f(v[cid]['acc'])} | {_f(t[cid]['acc']) if cid in t else '-'} | "
                         f"{_f(v[cid].get('sqnr_db'))} | {_f(v[cid].get('avg_bits_payload'))} |")
    best = pick_lambda_best(val_rows)
    for arm in meta["arms"]:
        for target in meta["targets"]:
            key = (arm, float(target))
            lines += ["", f"## {arm}, target <= {target:g} bits/weight", "",
                      "| method | param | avg bits | +meta | layers8 | val acc % | test acc % | test vs E4M11 (pp) | SQNR dB |",
                      "|---|---|---|---|---|---|---|---|---|"]

            def row(label, cid):
                if cid not in v:
                    return
                tt = t.get(cid)
                d = (float(tt["acc"]) - float(base["acc"])) if (tt and base) else None
                lines.append(f"| {label} | {v[cid]['param']} | {_f(v[cid]['avg_bits_payload'])} | "
                             f"{_f(v[cid]['avg_bits_with_metadata'])} | {v[cid]['n_layers8']} | {_f(v[cid]['acc'])} | "
                             f"{_f(tt['acc']) if tt else '-'} | {('%+.2f' % d) if d is not None else '-'} | "
                             f"{_f(v[cid]['sqnr_db'])} |")
            if key in best:
                row("lambda (best on val)", best[key])
            for lam in (0.0, 1.0):
                cid = config_id(arm, target, "lambda", lam)
                row("Fisher only (lambda=0)" if lam == 0 else "error only (lambda=1)", cid)
            row("product (sum F*dw^2)", config_id(arm, target, "product", None))
            row("worst (max damage)", config_id(arm, target, "worst", None))
            rids = [r["config_id"] for r in val_rows
                    if r["arm"] == arm and r["method"] == "random" and float(r["target_avg_bits"]) == float(target)]
            if rids:
                va = np.array([float(v[c]["acc"]) for c in rids])
                ta = np.array([float(t[c]["acc"]) for c in rids if c in t])
                bits = np.mean([float(v[c]["avg_bits_payload"]) for c in rids])
                tmean = f"{ta.mean():.2f} +/- {ta.std():.2f}" if len(ta) else "-"
                d = f"{ta.mean() - float(base['acc']):+.2f}" if len(ta) and base else "-"
                lines.append(f"| random (n={len(rids)}) | - | {bits:.2f} | - | - | {va.mean():.2f} +/- {va.std():.2f} "
                             f"| {tmean} | {d} | - |")
            if key in best:
                lines += ["", f"lambda chosen on validation: {v[best[key]]['param']}; 8-bit layers: "
                              f"{', '.join(selections[best[key]]['layers8']) or '(none)'}"]
    return "\n".join(lines) + "\n"
