"""Upload a file from a GPU job to s3://<bucket>/shreyas-gpu/out/<key> via the runner's POST policy ($POST_JSON).
Files over 1 GB go up as <key>.partNNN pieces plus a <key>.done marker (same convention as upload_to_shreyas.py).

    python scripts/gpu/put.py runs/OW02/test_owner.parquet artifacts/OW02/test_owner.parquet
"""
import json
import os
import sys
from pathlib import Path

import requests

PART = 1024 ** 3
spec = json.loads(Path(os.environ["POST_JSON"]).read_text())
src, key = Path(sys.argv[1]), sys.argv[2]


def send(k: str, blob: bytes) -> None:
    r = requests.post(spec["url"], data={**spec["fields"], "key": f"shreyas-gpu/out/{k}"},
                      files={"file": (Path(k).name, blob)}, timeout=7200)
    r.raise_for_status()


size = src.stat().st_size
n = 0
with open(src, "rb") as fh:
    if size <= PART:
        send(key, fh.read())
        n = 1
    else:
        while blob := fh.read(PART):
            send(f"{key}.part{n:03d}", blob)
            n += 1
send(f"{key}.done", json.dumps({"bytes": size, "parts": n, "split": size > PART}).encode())
print(f"uploaded {src} -> shreyas-gpu/out/{key} ({size / 1e9:.2f} GB, {n} part(s))")
