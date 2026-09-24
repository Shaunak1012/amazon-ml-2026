"""Build the final code zip from git-tracked files only (no data, weights, or secrets).

    python scripts/make_submission_zip.py                       # zip HEAD's tracked files
    python scripts/make_submission_zip.py --predictions submissions/final.csv --sample data/sample.csv \
        --doc docs/APPROACH.pdf --name team_final

Refuses to run with uncommitted changes (the zip must match a commit) unless --allow-dirty.
Output: dist/<name>_<shortsha>.zip + MANIFEST.txt inside (commit, tag, file list).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAX_FILE_MB = 20
BLOCKED = (".env",)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="amazon_ml_2026_code")
    ap.add_argument("--predictions", help="predictions CSV to include (validated first)")
    ap.add_argument("--sample", help="sample submission for validation of --predictions")
    ap.add_argument("--doc", help="approach doc (PDF/MD/DOCX) to include at zip root")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out-dir", default="dist")
    a = ap.parse_args()

    if git("status", "--porcelain") and not a.allow_dirty:
        print("ERROR: uncommitted changes. Commit first (the zip must be reproducible from a commit).")
        return 1
    sha = git("rev-parse", "--short", "HEAD")
    tag = subprocess.run(["git", "describe", "--tags", "--exact-match"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip() or "(untagged)"
    files = [f for f in git("ls-files").splitlines() if f]

    bad = [f for f in files if Path(f).name in BLOCKED]
    big = [f for f in files if (ROOT / f).exists() and (ROOT / f).stat().st_size > MAX_FILE_MB * 2**20]
    if bad or big:
        print(f"ERROR: refusing to zip secrets/large files: {bad + big}")
        return 1

    if a.predictions:
        from src.submission import validate_submission

        rep = validate_submission(a.predictions, sample=a.sample)
        print(rep)
        if not rep.ok:
            return 1

    out = ROOT / a.out_dir / f"{a.name}_{sha}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = [f"commit: {git('rev-parse', 'HEAD')}", f"tag: {tag}",
                f"built: {datetime.now(timezone.utc).isoformat()}", "", *files]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if (ROOT / f).exists():
                z.write(ROOT / f, f"code/{f}")
        if a.predictions:
            z.write(a.predictions, Path(a.predictions).name)
        if a.doc:
            z.write(a.doc, Path(a.doc).name)
        z.writestr("MANIFEST.txt", "\n".join(manifest))
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 2**20:.2f} MB, {len(files)} files, {sha} {tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
