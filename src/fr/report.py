"""Paper-target comparison and Markdown rendering for the step-1 reports."""
from __future__ import annotations

import math
from typing import List, Optional

import yaml


def load_targets(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _fmt(x, nd=4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _val(t: dict, key: str):
    e = t.get(key) or {}
    return e.get("value"), e


def compare_to_targets(fp32: dict, q: dict, targets: dict, subset: Optional[int]) -> List[dict]:
    """Rows of {item, ours, paper, status, source}. Never pass/fail-tunes anything."""
    rows = []
    sub_note = f" (on first {subset} test images)" if subset else ""

    def add(item, ours, paper, status, source, note=""):
        rows.append({"item": item, "ours": ours, "paper": paper, "status": status,
                     "source": source, "note": note})

    pv, e = _val(targets, "fp32_accuracy_percent")
    ours = fp32["metrics"]["accuracy_percent"]
    add("FP32 accuracy (%)" + sub_note, ours, pv,
        "no paper value" if pv is None else f"delta {ours - pv:+.2f} pp", e.get("source", ""))

    pv, e = _val(targets, "e4m11_accuracy_percent")
    ours = q["metrics"]["accuracy_percent"]
    add("E4M11 accuracy (%)" + sub_note, ours, pv,
        "no paper value (figure only)" if pv is None else f"delta {ours - pv:+.2f} pp",
        e.get("source", ""))

    e = targets.get("e4m11_accuracy_drop_pp") or {}
    drop = q["accuracy_drop_pp"]
    lo, hi = e.get("range_min"), e.get("range_max")
    if e.get("value") is not None:
        status = f"delta {drop - e['value']:+.2f} pp"
    elif lo is not None and hi is not None:
        status = "within range" if lo <= drop <= hi else (
            "below range" if drop < lo else "above range")
    else:
        status = "no paper value"
    add("E4M11 accuracy drop (pp)" + sub_note, drop, e.get("value") if e.get("value") is not None
        else (f"{lo}-{hi} (all E>=4 formats)" if lo is not None else None), status,
        e.get("source", ""), e.get("note", ""))

    e = targets.get("e4m11_sqnr_db") or {}
    s = q["sqnr"]["overall"]["sqnr_db"]
    lo, hi = e.get("range_min"), e.get("range_max")
    status = "no paper value"
    if lo is not None and hi is not None:
        status = "within range" if lo <= s <= hi else (
            f"below range by {lo - s:.2f} dB" if s < lo else f"above range by {s - hi:.2f} dB")
    add("E4M11 SQNR, Conv+Linear weights (dB)", s,
        f"{lo}-{hi} (all 9 model-dataset pairs); R18-C10 = {e.get('value')}", status,
        e.get("source", ""))

    e = targets.get("weight_range") or {}
    st_cl = fp32["weight_stats"]["convlinear"]
    st_all = fp32["weight_stats"]["all_parameters"]
    for label, key in (("w_min", "min"), ("w_max", "max")):
        pv = e.get(label)
        add(f"{label} (Table 3), Conv+Linear weights", st_cl[key], pv,
            "no paper value" if pv is None else f"delta {st_cl[key] - pv:+.3f}", e.get("source", ""))
        add(f"{label} (Table 3), all parameters", st_all[key], pv,
            "no paper value" if pv is None else f"delta {st_all[key] - pv:+.3f}", e.get("source", ""),
            "Table 3 does not say which tensors it covers")
    return rows


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(report: dict) -> str:
    L = [f"# Step 1 report: {report['run_name']}", ""]
    L.append(f"* checkpoint: `{report['checkpoint']}`")
    L.append(f"* normalization: `{report['normalization']}`")
    L.append(f"* test images used: {report['test_total']}"
             + (f" (subset of {report['test_subset']})" if report.get("test_subset") else " (full test set)"))
    L.append(f"* device: {report['device']}, torch {report['env']['torch']}, git {report['env']['git']['commit'][:10]}")
    if report.get("test_subset"):
        L.append("* **Subset evaluation: accuracy numbers are NOT comparable to the paper; SQNR does not depend on test data.**")
    L.append("")
    fp = report["fp32"]
    L += ["## FP32", "",
          _md_table(["metric", "value"],
                    [["accuracy (%)", _fmt(fp["metrics"]["accuracy_percent"], 2)],
                     ["correct/total", f"{fp['metrics']['correct']}/{fp['metrics']['total']}"],
                     ["mean cross-entropy", _fmt(fp["metrics"]["cross_entropy"])]]), "",
          "Weight statistics (paper Table 3 style):", "",
          _md_table(["tensors", "count", "min", "max", "abs max", "mean", "std"],
                    [[k, v["count"], _fmt(v["min"], 3), _fmt(v["max"], 3), _fmt(v["abs_max"], 3),
                      _fmt(v["mean"], 5), _fmt(v["std"], 5)] for k, v in fp["weight_stats"].items()]), ""]
    for mode, q in report["e4m11"].items():
        L += [f"## E4M11, mode=`{mode}`, scope=`{q['scope']}`, all-ones decode=`{q['all_ones']}`", "",
              _md_table(["metric", "FP32", "E4M11"],
                        [["accuracy (%)", _fmt(fp["metrics"]["accuracy_percent"], 2),
                          _fmt(q["metrics"]["accuracy_percent"], 2)],
                         ["mean cross-entropy", _fmt(fp["metrics"]["cross_entropy"]),
                          _fmt(q["metrics"]["cross_entropy"])]]), "",
              f"* accuracy drop: **{q['accuracy_drop_pp']:+.2f} pp**",
              f"* SQNR (Eq. 6, all {q['sqnr']['n_layers']} Conv+Linear weight tensors, "
              f"{q['sqnr']['overall']['count']:,} weights): **{_fmt(q['sqnr']['overall']['sqnr_db'], 4)} dB**"]
        if "sqnr_all_quantized_params" in q:
            a = q["sqnr_all_quantized_params"]
            L.append(f"* SQNR over all quantized parameters ({a['count']:,} values): {_fmt(a['sqnr_db'], 4)} dB")
        L += ["", "SQNR by |w| bin (original weight magnitude):", "",
              _md_table(["bin", "count", "% weights", "% signal power", "% error power", "SQNR (dB)"],
                        [[b["name"], f"{b['count']:,}", _fmt(100 * b["fraction_of_weights"], 2),
                          _fmt(100 * b["fraction_of_signal_power"], 3),
                          _fmt(100 * b["fraction_of_error_power"], 3), _fmt(b["sqnr_db"], 2)]
                         for b in q["sqnr"]["breakdown"]]), "",
              "Per-layer SQNR:", "",
              _md_table(["layer", "weights", "% abs(w)<2^-6", "abs max", "std", "SQNR (dB)"],
                        [[n, f"{v['count']:,}", _fmt(100 * v["frac_abs_lt_2^-6"], 1),
                          _fmt(v["stats"]["abs_max"], 3), _fmt(v["stats"]["std"], 5),
                          _fmt(v["sqnr_db"], 2)] for n, v in q["sqnr"]["per_layer"].items()]), ""]
        rows = report["paper_targets_comparison"].get(mode, [])
        if rows:
            L += ["Comparison with paper targets (`configs/paper_targets.yaml`; reported, not tuned):", "",
                  _md_table(["item", "ours", "paper", "status", "source"],
                            [[r["item"], _fmt(r["ours"], 4), _fmt(r["paper"], 4) if not isinstance(r["paper"], str)
                              else r["paper"], r["status"], r["source"]] for r in rows]), ""]
    return "\n".join(L)
