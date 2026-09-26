"""Pack the small stage-1/stage-2 artefacts Shreyas's CPU box needs and upload them to S3 via presigned URLs.

Run from the repo root on the GPU box (no AWS account needed; the URLs are write-only for fixed keys):
    python scripts/export_for_shreyas.py --urls urls.json

Exports (never embeddings or models; a few GB total):
  dev_stage1.parquet      train-chunk rows of the 100k dev S1 (+ ce_score, ce_score_2 ... from each CE dir)
  test_chunks/NNN.parquet every test chunk (all stage-1 features) + the CE columns
  stage2/<run>/...        dev_stage2_*.parquet, test_probs_stage2/3.parquet, stage2.json, predict.json
  folds_s1_k5.parquet
Then tar -> 4 GB parts -> HTTP PUT to the presigned URLs (part-00, part-01, ...).
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path

import pandas as pd

CE_DIRS = ["runs/E015-ce", "runs/E016-ce"]      # order = ce_score, ce_score_2 (as in the E016/E017 stage 2)
EXP = Path("runs/E015")
PART = 4 * 1024 ** 3


def with_ce(df: pd.DataFrame, chunk_name: str, split: str) -> pd.DataFrame:
    """Attach every available CE score for this chunk (row-aligned; checked on ids)."""
    for i, d in enumerate(CE_DIRS):
        f = Path(d) / f"{split}_ce" / chunk_name
        if not f.exists():
            continue
        ce = pd.read_parquet(f)
        full = pd.read_parquet(EXP / f"{split}_chunks" / chunk_name, columns=["s1_id", "cand_id"])
        assert len(ce) == len(full) and (ce.cand_id.values == full.cand_id.values).all(), f"misaligned {f}"
        ce.index = full.index
        col = "ce_score" if i == 0 else f"ce_score_{i + 1}"
        df[col] = ce.loc[df.index, "ce_score"].to_numpy()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help='json {"parts": [presigned PUT URLs for part-00, ...], "done": URL}')
    ap.add_argument("--out", default="runs/export_shreyas")
    a = ap.parse_args()
    spec = json.loads(Path(a.urls).read_text())
    urls = spec["parts"]
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "test_chunks").mkdir(parents=True)
    t0 = time.time()

    folds = pd.read_parquet("data/cache/folds_s1_k5.parquet")
    folds.to_parquet(out / "folds_s1_k5.parquet", index=False)
    dev = set(folds.loc[folds.dev, "s1_id"])
    parts = []
    for p in sorted((EXP / "train_chunks").glob("*.parquet")):
        c = pd.read_parquet(p)
        parts.append(with_ce(c[c.s1_id.isin(dev)], p.name, "train"))
        print(f"train {p.name}: {len(parts[-1]):,} dev rows [{time.time() - t0:.0f}s]", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(out / "dev_stage1.parquet", index=False, compression="zstd")
    for p in sorted((EXP / "test_chunks").glob("*.parquet")):
        with_ce(pd.read_parquet(p), p.name, "test").to_parquet(out / "test_chunks" / p.name, index=False,
                                                                compression="zstd")
        print(f"test {p.name} [{time.time() - t0:.0f}s]", flush=True)
    for d in [EXP, *sorted(x for x in EXP.iterdir() if x.is_dir() and x.name.startswith("sub"))]:
        files = [f for pat in ("dev_stage2_*.parquet", "dev_stage3_*.parquet", "test_probs_stage*.parquet",
                               "stage1.json", "stage2.json", "predict.json") for f in d.glob(pat)]
        if files:
            dst = out / "stage2" / d.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dst / f.name)
    tar = out.with_suffix(".tar")
    with tarfile.open(tar, "w") as tf:
        tf.add(out, arcname="export")
    size = tar.stat().st_size
    n = (size + PART - 1) // PART
    print(f"tar {size / 1e9:.2f} GB -> {n} part(s) [{time.time() - t0:.0f}s]", flush=True)
    if n > len(urls):
        raise SystemExit(f"need {n} URLs, have {len(urls)}")
    import requests
    with open(tar, "rb") as fh:
        for i in range(n):
            blob = fh.read(PART)
            r = requests.put(urls[i], data=blob, timeout=3600)
            r.raise_for_status()
            print(f"uploaded part {i:02d} ({len(blob) / 1e9:.2f} GB) [{time.time() - t0:.0f}s]", flush=True)
    done = requests.put(spec["done"], data=json.dumps({"parts": int(n), "bytes": size}).encode(), timeout=600)
    done.raise_for_status()
    print("EXPORT DONE")


if __name__ == "__main__":
    main()
