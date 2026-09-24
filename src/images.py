"""Robust, resumable bulk image downloader (for datasets that ship image URLs).

    python -m src.images --csv data/train.csv --url-col image_link --id-col sample_id \
        --out data/images/train --max-side 512 --workers 64

- threaded HTTP with connection pooling, retries + exponential backoff (429/5xx/timeouts)
- verifies every image decodes; converts to RGB JPEG; optional downscale (saves disk + dataloader time)
- resumable: files already on disk are skipped, so re-running after a crash only fetches what's missing
- writes <out>/manifest.csv: id, url, path, status (ok|cached|http_404|bad_image|error...), bytes, width, height
- NEVER silently drops rows: failed ids stay in the manifest so features can carry an explicit "missing image" flag
"""
from __future__ import annotations

import argparse
import hashlib
import io
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
RETRY_STATUS = {429, 500, 502, 503, 504}
_local = threading.local()


def _session(workers: int) -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = UA
        s.mount("http://", HTTPAdapter(pool_connections=workers, pool_maxsize=workers))
        s.mount("https://", HTTPAdapter(pool_connections=workers, pool_maxsize=workers))
        _local.s = s
    return s


def safe_name(key: str) -> str:
    """Filesystem-safe name; ids with odd characters (or raw URLs) get hashed."""
    k = str(key)
    if k and all(c.isalnum() or c in "-_." for c in k) and len(k) < 120:
        return k
    return hashlib.sha1(k.encode()).hexdigest()


def _save(content: bytes, path: Path, max_side: int | None, quality: int) -> tuple[int, int]:
    with Image.open(io.BytesIO(content)) as im:
        im.load()  # force full decode: truncated files fail here, not in the dataloader
        im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
        tmp = path.with_suffix(".part")
        im.save(tmp, "JPEG", quality=quality)
        tmp.replace(path)  # atomic: never leaves a half-written .jpg behind
        return im.size


def fetch_one(key: str, url: str, out: Path, max_side: int | None, quality: int, timeout: float,
              retries: int, workers: int) -> dict:
    path = out / f"{safe_name(key)}.jpg"
    rec = {"id": key, "url": url, "path": str(path), "status": "", "bytes": 0, "width": 0, "height": 0}
    if path.exists() and path.stat().st_size > 0:
        rec["status"] = "cached"
        return rec
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        rec["status"] = "no_url"
        return rec
    for attempt in range(retries + 1):
        try:
            r = _session(workers).get(url, timeout=timeout)
            if r.status_code in RETRY_STATUS and attempt < retries:
                wait = float(r.headers.get("Retry-After", 0) or 0) or (2 ** attempt + random.random())
                time.sleep(min(wait, 30))
                continue
            if r.status_code != 200:
                rec["status"] = f"http_{r.status_code}"
                return rec
            rec["bytes"] = len(r.content)
            rec["width"], rec["height"] = _save(r.content, path, max_side, quality)
            rec["status"] = "ok"
            return rec
        except (UnidentifiedImageError, OSError, ValueError) as e:
            if isinstance(e, requests.RequestException) and attempt < retries:
                time.sleep(2 ** attempt + random.random())
                continue
            rec["status"] = "bad_image" if not isinstance(e, requests.RequestException) else f"error_{type(e).__name__}"
            return rec
    rec["status"] = rec["status"] or "error_retries"
    return rec


def download_images(ids, urls, out_dir: str | Path, max_side: int | None = 512, workers: int = 64,
                    timeout: float = 15.0, retries: int = 3, quality: int = 92, progress: bool = True) -> pd.DataFrame:
    """Download all (id, url) pairs; returns the manifest (one row per input, same order)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = list(zip(map(str, ids), urls))
    results: list[dict | None] = [None] * len(pairs)
    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=len(pairs), unit="img", smoothing=0.05)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, k, u, out, max_side, quality, timeout, retries, workers): i
                for i, (k, u) in enumerate(pairs)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    man = pd.DataFrame(results)
    man.to_csv(out / "manifest.csv", index=False)
    counts = man["status"].value_counts().to_dict()
    good = int(man["status"].isin(["ok", "cached"]).sum())
    print(f"{good}/{len(man)} images available ({good / max(len(man), 1):.2%}) in {time.time() - t0:.0f}s | {counts}")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.images")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--url-col", required=True)
    ap.add_argument("--id-col", help="default: use the URL itself (hashed) as the file name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-side", type=int, default=512, help="0 = keep original size")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=15)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, help="only the first N rows (smoke test)")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.csv, dtype=str, keep_default_na=False, nrows=a.limit)
    ids = df[a.id_col] if a.id_col else df[a.url_col]
    man = download_images(ids, df[a.url_col], a.out, a.max_side or None, a.workers, a.timeout, a.retries)
    return 0 if man["status"].isin(["ok", "cached"]).any() else 1


if __name__ == "__main__":
    sys.exit(main())
