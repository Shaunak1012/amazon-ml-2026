"""Pack the stage-2 artefacts the CPU box (SageMaker, shreyas branch) needs and upload them via presigned S3 URLs.

Run from the repo root on the GPU box once `er_fullpass stage2 --frames <dir>` has cached both frames
(E019 does: runs/frames/E019). No AWS account needed; the URLs are write-only for fixed keys.
    python runs/export_for_shreyas.py --urls runs/urls_shreyas.json [--frames runs/frames/E019]

Exports (no embeddings, no model weights):
  frames/                 train_frame.parquet, test_frame.parquet, frames.json (as cached)
  train_min.parquet       [s1_id, cand_id, prob, similarity cols] of ALL train chunk rows (population features)
  stage2/<run>/...        dev_stage2_*.parquet, test_probs_stage2.parquet, stage2.json, predict.json
Then tar -> 4 GB parts -> HTTP PUT to the presigned URLs; a DONE marker last.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

EXP = Path("runs/E015")
MIN_COLS = ["s1_id", "cand_id", "prob", "cos_name", "name_ratio", "name_jw", "name_full_tset", "name_tsort"]
PART = 4 * 1024 ** 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help='json {"parts": [presigned PUT URLs for part-00, ...], "done": URL}')
    ap.add_argument("--frames", default="runs/frames/E019")
    ap.add_argument("--out", default="runs/export_shreyas")
    a = ap.parse_args()
    spec = json.loads(Path(a.urls).read_text())
    frames = Path(a.frames)
    if not (frames / "train_frame.parquet").exists():
        raise SystemExit(f"{frames}/train_frame.parquet missing: run stage 2 with --frames {frames} first")
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.time()

    shutil.copy2("data/cache/folds_s1_k5.parquet", out / "folds_s1_k5.parquet")
    files = sorted((EXP / "train_chunks").glob("*.parquet"))
    cols = [c for c in MIN_COLS if c in pq.read_schema(files[0]).names]
    parts = [pd.read_parquet(p, columns=cols) for p in files]
    pd.concat(parts, ignore_index=True).to_parquet(out / "train_min.parquet", index=False, compression="zstd")
    print(f"train_min: {sum(map(len, parts)):,} rows, cols {list(parts[0].columns)} [{time.time() - t0:.0f}s]", flush=True)
    del parts
    for d in [EXP, *sorted(x for x in EXP.iterdir() if x.is_dir() and x.name.startswith(("sub", "E0")))]:
        files = [f for pat in ("dev_stage2_*.parquet", "test_probs_stage*.parquet", "stage2.json", "predict.json")
                 for f in d.glob(pat)]
        if files:
            dst = out / "stage2" / d.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dst / f.name)
    tar = out.with_suffix(".tar")
    with tarfile.open(tar, "w") as tf:
        tf.add(out, arcname="export")
        tf.add(frames, arcname="export/frames")
    size = tar.stat().st_size
    n = (size + PART - 1) // PART
    print(f"tar {size / 1e9:.2f} GB -> {n} part(s) [{time.time() - t0:.0f}s]", flush=True)
    if n > len(spec["parts"]):
        raise SystemExit(f"need {n} URLs, have {len(spec['parts'])}")
    import requests
    with open(tar, "rb") as fh:
        for i in range(n):
            blob = fh.read(PART)
            requests.put(spec["parts"][i], data=blob, timeout=7200).raise_for_status()
            print(f"uploaded part {i:02d} ({len(blob) / 1e9:.2f} GB) [{time.time() - t0:.0f}s]", flush=True)
    requests.put(spec["done"], data=json.dumps({"parts": int(n), "bytes": size}).encode(), timeout=600).raise_for_status()
    print("EXPORT DONE")


if __name__ == "__main__":
    main()
