#!/usr/bin/env bash
# Hand-off in Shaunak's --ce-dir format: per-chunk parquets row-aligned with runs/E015/{train,test}_chunks,
# columns s1_id, cand_id, ce_score (NaN where our model did not score). One dir per feature:
#   OW04p (owner prob own_p), OW04m (owner margin own_margin), CE03 (e5-large CE ce3).
# Train chunks: the exact files he uploaded. Test chunks: rebuilt from the E020 test frame, whose rows are the test chunks
# concatenated in order (stage2_frames, keep_s1=None); chunk k = test S1 positions [200k*k, 200k*(k+1)).
set -euo pipefail
PY=.venv/bin/python
$PY - <<'PYEOF'
import numpy as np
import pandas as pd
from pathlib import Path
from src.er_data import cache_dir
out = Path("runs/handoff_chunks")
feats = {"OW04p": ("runs/OW04/{s}_owner.parquet", "own_p"), "OW04m": ("runs/OW04/{s}_owner.parquet", "own_margin"),
         "CE03": ("runs/CE03dev/{s}_feats.parquet", "ce3")}
src = {s: {k: pd.read_parquet(f.format(s=s), columns=["s1_id", "cand_id", c]).rename(columns={c: "ce_score"})
           for k, (f, c) in feats.items()} for s in ("train", "test")}
def write(split, name, chunk_df):
    for k, E in src[split].items():
        d = out / k / f"{split}_ce"
        d.mkdir(parents=True, exist_ok=True)
        m = chunk_df.merge(E, on=["s1_id", "cand_id"], how="left", validate="one_to_one")
        assert len(m) == len(chunk_df) and (m.cand_id.values == chunk_df.cand_id.values).all()
        m.to_parquet(d / name, index=False)
for p in sorted(Path("runs/shared/E015/train_chunks").glob("*.parquet")):
    write("train", p.name, pd.read_parquet(p, columns=["s1_id", "cand_id"]))
    print("train", p.name, flush=True)
T = pd.read_parquet("runs/shared/frames/E020/test_frame.parquet", columns=["s1_id", "cand_id"])
order = pd.read_parquet(cache_dir() / "test_s1.parquet", columns=["entity_id"]).entity_id
pos = pd.Series(np.arange(len(order)), index=order.values).reindex(T.s1_id).to_numpy()
ck = pos // 200_000
assert (np.diff(ck) >= 0).all(), "test frame rows are not in chunk order"
for c in np.unique(ck):
    part = T[ck == c].reset_index(drop=True)
    write("test", f"{c:03d}.parquet", part)
    print("test", f"{c:03d}", len(part), "rows", part.s1_id.nunique(), "S1", flush=True)
for k in feats:
    for s in ("train", "test"):
        cov = pd.concat([pd.read_parquet(f) for f in sorted((out / k / f"{s}_ce").glob("*.parquet"))]).ce_score.notna().mean()
        print(f"{k} {s}: coverage {cov:.3%}")
PYEOF
cd runs/handoff_chunks && for k in OW04p OW04m CE03; do tar -cf ../handoff_$k.tar $k; done && cd ../..
for k in OW04p OW04m CE03; do aws s3 cp --only-show-errors runs/handoff_$k.tar "s3://$S3_BUCKET/$S3_PREFIX/artifacts/handoff/handoff_$k.tar"; done
ls -la runs/handoff_*.tar
