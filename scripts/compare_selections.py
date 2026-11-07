#!/usr/bin/env python
"""Compare two saved layer selections (for example Fisher from training images vs Fisher from held-out images).

  python scripts/compare_selections.py $FR_ROOT/results/scratch/mixed $FR_ROOT/results/scratch/mixed_heldout

For every config id present in both `selections.json` files (random draws are skipped: they do not depend on the
Fisher) prints the low-bit layers of each and the Jaccard overlap counted in weights (1 = identical sets).
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from fr.stability import weighted_jaccard


def compare(dir_a: Path, dir_b: Path) -> str:
    sa, sb = json.load(open(dir_a / "selections.json")), json.load(open(dir_b / "selections.json"))
    meta = json.load(open(dir_a / "sweep_meta.json"))
    size_of = dict(zip(meta["layers"], meta["layer_sizes"]))
    short = lambda names: ", ".join(n.replace(".weight", "") for n in names) or "(none)"
    lines = [f"# Selection comparison", "", f"* A = `{dir_a}`", f"* B = `{dir_b}`",
             "* Jaccard counted in weights; random draws omitted (they do not depend on the Fisher)", "",
             "| config | Jaccard | layers A | layers B |", "|---|---|---|---|"]
    js = []
    for cid in sa:
        if cid not in sb or "|random|" in cid:
            continue
        j = weighted_jaccard(sa[cid]["layers8"], sb[cid]["layers8"], size_of)
        js.append(j)
        lines.append(f"| {cid} | {j:.3f} | {short(sa[cid]['layers8'])} | {short(sb[cid]['layers8'])} |")
    if js:
        lines += ["", f"mean Jaccard over {len(js)} configurations: {sum(js) / len(js):.3f}; "
                      f"identical selections: {sum(1 for j in js if j == 1.0)}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a", type=Path)
    ap.add_argument("dir_b", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    md = compare(a.dir_a, a.dir_b)
    print(md)
    if a.out:
        a.out.write_text(md, encoding="utf-8")
        print("written:", a.out)


if __name__ == "__main__":
    main()
