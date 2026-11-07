"""Copy the small result files and the model checkpoint of a run into tracked folders for sharing.

    python scripts/export_results.py --fr-root $FR_ROOT --run-name scratch

Copies  <fr-root>/results/<run>/**   ->  reports/<run>/**     (every file <= --max-report-mb, default 5 MB)
        <fr-root>/checkpoints/<run>/best.pth -> models/<run>/best.pth   (if <= --max-model-mb, default 90 MB;
                                                                         GitHub rejects files over 100 MB)
Larger files are listed in reports/<run>/EXPORT_MANIFEST.md as skipped, with their size, and are NOT copied.
Nothing is deleted or modified in <fr-root>. The repo .gitignore ignores results/, checkpoints/ and *.pth, so the
script prints the `git add -f` command to run afterwards.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fr-root", required=True, type=Path)
    ap.add_argument("--run-name", default="scratch")
    ap.add_argument("--max-report-mb", type=float, default=5.0)
    ap.add_argument("--max-model-mb", type=float, default=90.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = a.fr_root / "results" / a.run_name
    if not src.is_dir():
        raise SystemExit(f"not found: {src}")
    dst = REPO / "reports" / a.run_name
    copied, skipped = [], []
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = f.relative_to(src)
        mb = f.stat().st_size / 2**20
        if mb <= a.max_report_mb:
            copied.append((rel.as_posix(), mb))
            if not a.dry_run:
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst / rel)
        else:
            skipped.append((rel.as_posix(), mb, "larger than --max-report-mb"))

    model_lines = []
    ck = a.fr_root / "checkpoints" / a.run_name / "best.pth"
    if ck.is_file():
        mb = ck.stat().st_size / 2**20
        if mb <= a.max_model_mb:
            model_lines.append(f"- `models/{a.run_name}/best.pth`  {mb:.1f} MB  sha256[:12]={sha(ck)}")
            if not a.dry_run:
                (REPO / "models" / a.run_name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(ck, REPO / "models" / a.run_name / "best.pth")
        else:
            skipped.append((f"checkpoints/{a.run_name}/best.pth", mb, "larger than --max-model-mb"))
    else:
        model_lines.append(f"- checkpoint not found at {ck}")

    L = [f"# Export manifest for run `{a.run_name}`", "",
         "## Model", *model_lines, "",
         f"## Reports copied to `reports/{a.run_name}/` ({len(copied)} files, {sum(m for _, m in copied):.2f} MB)", ""]
    L += [f"- `{r}`  {m:.3f} MB" for r, m in copied]
    L += ["", f"## Skipped ({len(skipped)})", ""]
    L += [f"- `{r}`  {m:.1f} MB  ({why})" for r, m, why in skipped] or ["- none"]
    text = "\n".join(L) + "\n"
    print(text)
    if not a.dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "EXPORT_MANIFEST.md").write_text(text, encoding="utf-8")
        print(f"\nNext:\n  git add -f reports models\n  git commit -m 'Add experiment results and baseline checkpoint'\n  git push")


if __name__ == "__main__":
    main()
