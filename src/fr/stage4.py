"""J-FAMP runbook Stage 4: global and layer-wise fault screening (Sections 6.1 - 6.3).

Everything here sits on top of the validated Stage 3 injector (``fr.stage3``): faults are applied to the STORED ENCODED bytes.

Configurations (one "configuration" = format x scope x fault group x k, run for ``trials`` independent trials):

  global      scope = all quantizable weights (random encoded weights across all 21 layers)      k in {1, 8}
  layerwise   scope = one layer at a time, all other layers fixed                               k in {1, 8}
  scale ext.  (extension, INT formats only) scope = the float32 per-tensor scale words: global (21 words, k in {1, 8}) and
              one scale word per layer (k = 1 only: a layer has a single scale word, k = 8 distinct words is impossible)

Groups: floating point formats sign / exponent / fraction, integer formats sign / value (NO exponent group), plus ``all``
(every bit of the element). Scale scope: float32 sign / exponent / fraction / all.

Resumable: every trial has a unique id ``format|scope_kind|scope_name|group|k<k>|t<trial>``; records are appended to
``fault_trials.jsonl`` as they finish; a restart skips finished ids; ``verify_completeness`` checks that no trial is duplicated
or missing. Summaries are regenerated from the raw records only.
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import stage3 as S3

SCHEMA_VERSION = "stage4-trial-v1"
K_VALUES = (1, 8)
FLAG_NAMES = ("weights_nonfinite", "activations_nonfinite", "logits_nonfinite", "loss_nonfinite")


def trial_id(fmt: str, scope_kind: str, scope_name: str, group: str, k: int, trial: int) -> str:
    return f"{fmt}|{scope_kind}|{scope_name}|{group}|k{k}|t{trial}"


# ------------------------------------------------------------------------------------------------ configurations
def enumerate_configs(fmt: str, layer_names: Sequence[str], include_scales: bool = True,
                      layers: Optional[Sequence[int]] = None) -> Tuple[List[dict], List[dict]]:
    """(configurations, skipped) for one format. ``layer_names`` are module paths (``layer1.0.conv1``)."""
    cfgs, skipped = [], []
    all_layers = list(range(len(layer_names)))
    lay = all_layers if layers is None else list(layers)
    kind = S3.SPECS[fmt]["kind"]
    for g in S3.format_groups(fmt):
        if not S3.group_bits(fmt, g):
            skipped.append({"format": fmt, "group": g, "reason": "group does not exist in this format"})
            continue
        for k in K_VALUES:
            cfgs.append({"format": fmt, "level": "global", "scope_kind": "weights", "scope_name": "GLOBAL",
                         "scope_layers": all_layers, "group": g, "k": k})
        for l in lay:
            for k in K_VALUES:
                cfgs.append({"format": fmt, "level": "layerwise", "scope_kind": "weights", "scope_name": layer_names[l],
                             "scope_layers": [l], "group": g, "k": k})
    if kind == "int" and not S3.group_bits(fmt, "exponent"):
        skipped.append({"format": fmt, "group": "exponent", "reason": "integer format has no exponent field (runbook 6.1)"})
    if include_scales and fmt in S3.SCALE_FORMATS:
        for g in S3.format_groups(fmt, "scales"):
            for k in K_VALUES:
                cfgs.append({"format": fmt, "level": "global", "scope_kind": "scales", "scope_name": "GLOBAL-SCALES",
                             "scope_layers": all_layers, "group": g, "k": k})
            for l in lay:
                cfgs.append({"format": fmt, "level": "layerwise", "scope_kind": "scales", "scope_name": layer_names[l],
                             "scope_layers": [l], "group": g, "k": 1})
            skipped.append({"format": fmt, "scope_kind": "scales", "level": "layerwise", "group": g, "k": 8,
                            "reason": "a layer has a single scale word; k = 8 distinct scale words cannot be drawn"})
    return cfgs, skipped


def expected_ids(cfgs: Iterable[dict], trials: int) -> List[str]:
    return [trial_id(c["format"], c["scope_kind"], c["scope_name"], c["group"], c["k"], t) for c in cfgs for t in range(trials)]


# ------------------------------------------------------------------------------------------------ trial log (resumable)
class TrialLog:
    """Append-only JSONL of trial records with duplicate detection and resume."""

    def __init__(self, path):
        self.path = Path(path)
        self.done: Dict[str, dict] = {}
        self.duplicates: List[str] = []
        if self.path.exists():
            with open(self.path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec["trial_id"] in self.done:
                        self.duplicates.append(rec["trial_id"])
                    self.done[rec["trial_id"]] = rec
        self._fh = open(self.path, "a")

    def has(self, tid: str) -> bool:
        return tid in self.done

    def append(self, rec: dict) -> None:
        if rec["trial_id"] in self.done:
            raise RuntimeError(f"duplicate trial id {rec['trial_id']}")
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        self.done[rec["trial_id"]] = rec

    def close(self) -> None:
        self._fh.close()


def load_records(path) -> List[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def verify_completeness(records: Sequence[dict], expected: Sequence[str]) -> dict:
    from collections import Counter
    ids = [r["trial_id"] for r in records]
    dup = sorted(i for i, c in Counter(ids).items() if c > 1)
    exp = set(expected)
    got = set(ids)
    return {"expected": len(exp), "recorded": len(got), "duplicates": dup, "missing": sorted(exp - got)[:50],
            "n_missing": len(exp - got), "unexpected": sorted(got - exp)[:50], "n_unexpected": len(got - exp),
            "complete": not dup and not (exp - got) and not (got - exp)}


# ------------------------------------------------------------------------------------------------ records
def compact_record(rec: dict, tid: str, level: str) -> dict:
    """Section 5.3 record, flips stored as [target, layer_id, element, bit_index] to keep the file small."""
    r = {k: v for k, v in rec.items() if k not in ("sampling", "_logits")}
    r["flips"] = [[f["target"][0], f["layer_id"], f["element"], f["bit_index"]] for f in rec["flips"]]
    r["trial_id"] = tid
    r["level"] = level
    r["schema"] = SCHEMA_VERSION
    return r


def run_configs(inj: S3.EncodedFaultInjector, X, y, clean: dict, ctx: dict, cfgs: Sequence[dict], trials: int, base_seed: int,
                log: TrialLog, progress=print) -> dict:
    """Run every (configuration, trial) not yet in ``log``. A restoration failure aborts the campaign at once."""
    n_new = n_skip = 0
    for ci, c in enumerate(cfgs):
        new_here = []
        for t in range(trials):
            tid = trial_id(c["format"], c["scope_kind"], c["scope_name"], c["group"], c["k"], t)
            if log.has(tid):
                n_skip += 1
                continue
            rec = S3.run_trial(inj, X, y, clean, ctx, fmt=c["format"], scope_layers=c["scope_layers"], scope_name=c["scope_name"],
                               scope_kind=c["scope_kind"], group=c["group"], k=c["k"], trial=t, base_seed=base_seed)
            if not rec["restored_exactly"] or not rec["flip_applied_to_encoded_bytes"]:
                raise RuntimeError(f"injector integrity failure in trial {tid}: restored={rec['restored_exactly']} "
                                   f"applied={rec['flip_applied_to_encoded_bytes']}")
            log.append(compact_record(rec, tid, c["level"]))
            new_here.append(rec["fault_accuracy"])
            n_new += 1
        if new_here and (c["level"] == "global" or ci % 8 == 0):
            progress(f"  [{c['format']}|{c['scope_kind']}|{c['scope_name']}|{c['group']}|k{c['k']}] {len(new_here)} trials, "
                     f"mean fault accuracy {np.mean(new_here):.2f} % (clean {clean['accuracy']:.2f} %)")
    return {"new": n_new, "skipped_already_done": n_skip}


# ------------------------------------------------------------------------------------------------ summaries
SUMMARY_FIELDS = ["layer_id", "layer_name", "format", "scope_kind", "fault_group", "k", "n_trials", "sample_count",
                  "clean_accuracy", "clean_loss", "fault_accuracy_mean",
                  "drop_pp_mean", "drop_pp_median", "drop_pp_std", "drop_pp_p90", "drop_pp_worst", "drop_pp_sem",
                  "rel_drop_percent_mean", "loss_increase_mean", "loss_increase_median", "loss_increase_p90",
                  "loss_increase_worst", "n_loss_finite",
                  "failures", "failure_rate", "fail_weights_nonfinite", "fail_activations_nonfinite", "fail_logits_nonfinite",
                  "fail_loss_nonfinite"]


def _stats(a: np.ndarray) -> dict:
    if a.size == 0:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "p90": math.nan, "worst": math.nan}
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if a.size > 1 else math.nan,
            "p90": float(np.percentile(a, 90)), "worst": float(a.max())}


def summarize(records: Sequence[dict], names: Optional[Sequence[str]] = None) -> List[dict]:
    """One row per (layer scope x format x scope kind x group x k), computed from the raw trial records only.

    drop_pp = clean accuracy - fault accuracy (percentage points; images with non-finite logits count as incorrect);
    std uses ddof = 1; p90 = 90th percentile (linear interpolation) of the drop; worst = maximum drop;
    loss_increase = fault loss - clean loss over trials with a finite fault loss (n_loss_finite of n_trials).
    """
    groups: Dict[tuple, list] = defaultdict(list)
    for r in records:
        groups[(r["layer_id"], r["layer_name"], r["format"], r["scope_kind"], r["fault_group"], r["requested_k"])].append(r)
    rows = []
    for key, rs in groups.items():
        layer_id, layer_name, fmt, kind, grp, k = key
        drop = np.array([r["clean_accuracy"] - r["fault_accuracy"] for r in rs])
        clean = rs[0]["clean_accuracy"]
        lossinc = np.array([r["fault_loss"] - r["clean_loss"] for r in rs if r["fault_loss"] is not None])
        flags = [set(r["output_flag"].split(",")) - {"none"} for r in rs]
        d, li = _stats(drop), _stats(lossinc)
        n = len(rs)
        rows.append({"layer_id": layer_id, "layer_name": layer_name, "format": fmt, "scope_kind": kind, "fault_group": grp, "k": k,
                     "n_trials": n, "sample_count": rs[0]["sample_count"], "clean_accuracy": clean, "clean_loss": rs[0]["clean_loss"],
                     "fault_accuracy_mean": float(np.mean([r["fault_accuracy"] for r in rs])),
                     "drop_pp_mean": d["mean"], "drop_pp_median": d["median"], "drop_pp_std": d["std"], "drop_pp_p90": d["p90"],
                     "drop_pp_worst": d["worst"], "drop_pp_sem": d["std"] / math.sqrt(n) if n > 1 else math.nan,
                     "rel_drop_percent_mean": 100.0 * d["mean"] / clean if clean else math.nan,
                     "loss_increase_mean": li["mean"], "loss_increase_median": li["median"], "loss_increase_p90": li["p90"],
                     "loss_increase_worst": li["worst"], "n_loss_finite": int(lossinc.size),
                     "failures": sum(1 for f in flags if f), "failure_rate": sum(1 for f in flags if f) / n,
                     **{f"fail_{name}": sum(name in f for f in flags) for name in FLAG_NAMES}})
    rows.sort(key=lambda r: (r["format"], r["scope_kind"], r["layer_id"], r["fault_group"], r["k"]))
    return rows


def split_global(rows: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    g = [r for r in rows if r["layer_id"] == -1]
    return g, [r for r in rows if r["layer_id"] != -1]


def heatmap(rows: Sequence[dict], layer_meta: Sequence[dict]) -> Tuple[List[str], List[dict]]:
    """Layer x (format, scope kind, group, k) matrix of mean accuracy drop (pp) and p90 drop. Columns ``<fmt>|<kind>|<group>|k<k>|mean``."""
    layer_rows = [r for r in rows if r["layer_id"] != -1]
    cols = sorted({(r["format"], r["scope_kind"], r["fault_group"], r["k"]) for r in layer_rows})
    idx = {(r["layer_id"], r["format"], r["scope_kind"], r["fault_group"], r["k"]): r for r in layer_rows}
    header = ["layer_id", "layer_name", "num_weights"]
    for c in cols:
        header += [f"{c[0]}|{c[1]}|{c[2]}|k{c[3]}|mean_drop_pp", f"{c[0]}|{c[1]}|{c[2]}|k{c[3]}|p90_drop_pp"]
    out = []
    for m in layer_meta:
        row = {"layer_id": int(m["layer_id"]), "layer_name": m["module_name"], "num_weights": int(m["num_weights"])}
        for c in cols:
            r = idx.get((int(m["layer_id"]),) + c)
            row[f"{c[0]}|{c[1]}|{c[2]}|k{c[3]}|mean_drop_pp"] = r["drop_pp_mean"] if r else ""
            row[f"{c[0]}|{c[1]}|{c[2]}|k{c[3]}|p90_drop_pp"] = r["drop_pp_p90"] if r else ""
        out.append(row)
    return header, out
