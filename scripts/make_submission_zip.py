"""Build the final package in the exact layout the organisers require (docs/COMPETITION.md):

    <team>_submission.zip
    ├── output/matching_results.tsv, output/candidate_pairs.tsv
    ├── code/business_entity_resolution/{src/, configs/, scripts/, monitor/, README.md, requirements.txt, ...}
    └── Documentation_template.md   (our filled-in methodology; .pdf also accepted)

    python scripts/make_submission_zip.py --team <team_name> --outputs output/ \
        --doc docs/Documentation_template.md --test-dir data/dataset/test

- code comes from git-tracked files only (no data, weights, .env); refuses uncommitted changes unless --allow-dirty
- code/.../README.md is docs/REPRODUCE.md (exact end-to-end run instructions), not the repo README
- both TSVs are validated with src.er_submission (incl. matches ⊆ candidates) before zipping
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
CODE_PREFIX = "code/business_entity_resolution"
CODE_DIRS = ("src/", "configs/", "scripts/", "monitor/", "tests/")
CODE_FILES = ("requirements.txt", "requirements-core.txt", "pyproject.toml", ".env.example")
MAX_FILE_MB = 20


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name used in <team>_submission.zip")
    ap.add_argument("--outputs", default="output", help="folder with matching_results.tsv + candidate_pairs.tsv")
    ap.add_argument("--doc", required=True, help="filled-in Documentation_template.md (or .pdf)")
    ap.add_argument("--readme", default="docs/REPRODUCE.md", help="becomes code/.../README.md")
    ap.add_argument("--test-dir", help="test split dir for full validation (strongly recommended)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out-dir", default="dist")
    a = ap.parse_args()

    if git("status", "--porcelain", "--untracked-files=no") and not a.allow_dirty:
        print("ERROR: uncommitted changes. Commit first (the zip must be reproducible from a commit).")
        return 1
    outputs = Path(a.outputs)
    match, cand = outputs / "matching_results.tsv", outputs / "candidate_pairs.tsv"
    from src.er_submission import validate_outputs

    rep = validate_outputs(match, cand, test_dir=a.test_dir)
    print(rep)
    if not rep.ok:
        return 1
    for p in (Path(a.doc), Path(a.readme)):
        if not p.exists():
            print(f"ERROR: missing {p}")
            return 1

    files = [f for f in git("ls-files").splitlines()
             if f.startswith(CODE_DIRS) or f in CODE_FILES]
    big = [f for f in files if (ROOT / f).stat().st_size > MAX_FILE_MB * 2**20]
    if big or any(Path(f).name == ".env" for f in files):
        print(f"ERROR: refusing to zip secrets/large files: {big}")
        return 1

    sha = git("rev-parse", "--short", "HEAD")
    out = ROOT / a.out_dir / f"{a.team}_submission.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc_name = "Documentation_template" + Path(a.doc).suffix
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(match, "output/matching_results.tsv")
        z.write(cand, "output/candidate_pairs.tsv")
        for f in files:
            z.write(ROOT / f, f"{CODE_PREFIX}/{f}")
        z.write(a.readme, f"{CODE_PREFIX}/README.md")
        z.write(a.doc, doc_name)
        z.writestr(f"{CODE_PREFIX}/BUILD_INFO.txt",
                   f"commit: {git('rev-parse', 'HEAD')}\nbuilt: {datetime.now(timezone.utc).isoformat()}\n")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    required = ["output/matching_results.tsv", "output/candidate_pairs.tsv", f"{CODE_PREFIX}/README.md",
                f"{CODE_PREFIX}/requirements.txt", doc_name]
    missing = [r for r in required if r not in names] + ([] if any(n.startswith(f"{CODE_PREFIX}/src/") for n in names)
                                                         else [f"{CODE_PREFIX}/src/"])
    if missing:
        print(f"ERROR: zip is missing {missing}")
        return 1
    print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MB, {len(names)} entries, commit {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
