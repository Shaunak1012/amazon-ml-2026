"""Upload local run artefacts to Shreyas's S3 bucket through one presigned POST (no AWS account needed).

The POST policy only allows keys under shared/runs/ and expires after 7 days. Files over 1 GB are sent as
<name>.partNNN pieces (reassembled on the SageMaker box by scripts/sm/jobs/fetch_shared.sh); a small
<path>.done marker is written after each complete file.

    python runs/upload_to_shreyas.py --post runs/post_shreyas.json runs/frames/E019 runs/E015/dev_stage2_E019.parquet ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

PART = 1024 ** 3


def send(post: dict, key: str, blob: bytes) -> None:
    fields = {**post["fields"], "key": key}
    r = requests.post(post["url"], data=fields, files={"file": (Path(key).name, blob)}, timeout=7200)
    if r.status_code not in (200, 201, 204):
        raise SystemExit(f"{key}: HTTP {r.status_code} {r.text[:300]}")


def upload_file(post: dict, path: Path) -> None:
    rel = path.as_posix().removeprefix("runs/")
    key = f"shared/runs/{rel}"
    size = path.stat().st_size
    if size <= PART:
        send(post, key, path.read_bytes())
        n = 1
    else:
        with open(path, "rb") as fh:
            n = 0
            while blob := fh.read(PART):
                send(post, f"{key}.part{n:03d}", blob)
                n += 1
    send(post, f"{key}.done", json.dumps({"bytes": size, "parts": n, "split": size > PART}).encode())
    print(f"ok {rel} ({size / 1e9:.2f} GB, {n} part(s))", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", required=True, help="presigned POST json (url + fields)")
    ap.add_argument("paths", nargs="+", help="files or folders under runs/")
    a = ap.parse_args()
    post = json.loads(Path(a.post).read_text())
    for p in map(Path, a.paths):
        files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
        for f in files:
            upload_file(post, f)
    print("UPLOAD DONE")


if __name__ == "__main__":
    main()
