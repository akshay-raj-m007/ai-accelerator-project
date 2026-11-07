#!/usr/bin/env python
"""Summarise the scale-metadata fault experiment (all-INT8 per-channel vs per-tensor).

  python scripts/compare_scale_faults.py --root $FR_ROOT/results/scratch/mixed_faults_scale --out report.md

Expected layout under ROOT (written by scripts/run_scale_faults.sh), each a normal `mixed_fault_campaign.py` output dir
`<mode>/<sampling>/<layout>/`:
  off/   weights only (scales fault-free), sampling bits       on/    scales share the memory, sampling bits
  only/  every strike hits a scale word (damage per scale hit)  groups/ weight bit-group breakdown, scales fault-free
Prints (1) storage of the scales, (2) accuracy with scales off vs on and the paired difference, (3) damage per scale
hit by bit group, (4) the weight bit-group breakdown. Nothing is tuned or selected here.
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from fr import faults_mixed as FM


def _camps(root: Path, mode: str):
    d = root / mode
    out = []
    if not d.exists():
        return out
    for sampling in sorted(p for p in d.iterdir() if p.is_dir()):
        out += FM.collect(sorted(p for p in sampling.iterdir() if p.is_dir()))
    return out


def _cell(c, g, k):
    return [r for r in c["rows"] if r["group"] == g and r["k"] == k]


def _stats(rows):
    a = np.array([r["accuracy"] for r in rows])
    fin = sum(1 for r in rows if not r["finite"])
    return (f"{a.mean():.2f} | {np.median(a):.2f} | {np.quantile(a, 0.10):.2f} | {a.min():.2f} | {fin}")


def compare(root: Path) -> str:
    off, on, only, grp = (_camps(root, m) for m in ("off", "on", "only", "groups"))
    L = ["# Scale-metadata faults: all-INT8 per-channel vs per-tensor", ""]
    any_c = (off or on or only or grp)
    if not any_c:
        return "no campaigns found\n"
    meta0 = any_c[0]["meta"]
    L += [f"* images: `{meta0.get('split', 'test')}` split ({meta0['clean']['total']} images); scales are float32 words; "
          "see `fr/faults_mixed.py` for the fault model", ""]
    # ---- storage
    L += ["## Storage of the scale metadata", "",
          "| layout | scale words | scale bits | payload bits (weights) | scale bits / payload | extra bits per weight |",
          "|---|---|---|---|---|---|"]
    seen = set()
    for c in off + on + only:
        m = c["meta"]
        if c["slug"] in seen or not m.get("n_scale_words"):
            continue
        seen.add(c["slug"])
        payload = m["avg_bits_payload"] * m["buffer_weights"]
        L.append(f"| {c['slug']} | {m['n_scale_words']} | {m['scale_bits']} | {int(payload)} | "
                 f"{100 * m['scale_bits'] / payload:.4f} % | {m['scale_bits'] / m['buffer_weights']:.5f} |")
    # ---- off vs on
    L += ["", "## Realistic load: scales fault-free (off) vs sharing the memory (on)", "",
          "Bit-uniform sampling (`bits`): each strike lands on a uniformly chosen bit of the whole buffer. Columns: "
          "mean / median / p10 / min test accuracy %, number of non-finite trials; `scale hits` = mean number of "
          "strikes per trial that landed in scale words.", "",
          "| layout | group | k | mode | mean | median | p10 | min | non-finite | scale hits/trial |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(c["slug"], r["group"], r["k"]) for c in off + on for r in c["rows"]})
    by = {("off", c["slug"]): c for c in off} | {("on", c["slug"]): c for c in on}
    for slug, g, k in keys:
        for mode in ("off", "on"):
            c = by.get((mode, slug))
            rows = _cell(c, g, k) if c else []
            if rows:
                hits = np.mean([r.get("scale_hits", 0) for r in rows])
                L.append(f"| {slug} | {g} | {k} | {mode} | {_stats(rows)} | {hits:.2f} |")
    L += ["", "Paired difference (on minus off, same trial index and same weight strikes where kept); "
          "`hit trials` = trials with at least one scale strike:", "",
          "| layout | group | k | trials | mean diff (pp) | s.e. | hit trials | mean diff on hit trials (pp) | "
          "worst diff (pp) |", "|---|---|---|---|---|---|---|---|---|"]
    for slug, g, k in keys:
        a, b = by.get(("off", slug)), by.get(("on", slug))
        if not (a and b):
            continue
        ra = {r["trial"]: r["accuracy"] for r in _cell(a, g, k)}
        d, dh = [], []
        for r in _cell(b, g, k):
            if r["trial"] in ra:
                d.append(r["accuracy"] - ra[r["trial"]])
                if r.get("scale_hits", 0) > 0:
                    dh.append(d[-1])
        if d:
            d = np.array(d)
            L.append(f"| {slug} | {g} | {k} | {d.size} | {d.mean():+.3f} | {d.std(ddof=1) / np.sqrt(d.size):.3f} | "
                     f"{len(dh)} | {np.mean(dh) if dh else float('nan'):+.3f} | {d.min():+.2f} |")
    # ---- damage per scale hit
    L += ["", "## Damage per scale hit (every strike hits a scale word)", "",
          "| layout | group | k | mean | median | p10 | min | non-finite | trials with drop > 1 pp | drop > 10 pp |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(only, key=lambda c: c["slug"]):
        clean = c["meta"]["clean"]["accuracy_percent"]
        for g, k in sorted({(r["group"], r["k"]) for r in c["rows"]}):
            rows = _cell(c, g, k)
            a = np.array([r["accuracy"] for r in rows])
            L.append(f"| {c['slug']} | {g} | {k} | {_stats(rows)} | "
                     f"{100 * np.mean(clean - a > 1):.0f} % | {100 * np.mean(clean - a > 10):.0f} % |")
    # ---- weight groups
    L += ["", "## Weight bit groups (scales fault-free)", "",
          "| layout | group | k | mean | median | p10 | min | non-finite |", "|---|---|---|---|---|---|---|---|"]
    for c in sorted(grp, key=lambda c: c["slug"]):
        for g, k in sorted({(r["group"], r["k"]) for r in c["rows"]}):
            L.append(f"| {c['slug']} | {g} | {k} | {_stats(_cell(c, g, k))} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    md = compare(a.root)
    print(md)
    if a.out:
        a.out.write_text(md, encoding="utf-8")
        print("written:", a.out)


if __name__ == "__main__":
    main()
