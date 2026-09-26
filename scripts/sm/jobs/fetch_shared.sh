#!/usr/bin/env bash
# SageMaker box: wait for Shaunak's uploads (scripts/upload_to_shreyas.py -> s3://<bucket>/shared/runs/), download,
# reassemble split files, build train_min from the stage-1 train chunks, and lay out runs/import/export/ for SH01.
set -euo pipefail
S=s3://$S3_BUCKET/shared/runs
need=(frames/E019/train_frame.parquet frames/E019/frames.json data/cache/folds_s1_k5.parquet E015/train_chunks/011.parquet)
for k in "${need[@]}"; do until aws s3 ls "$S/$k.done" >/dev/null 2>&1; do sleep 60; done; echo "ready: $k"; done
mkdir -p runs/shared && aws s3 sync --only-show-errors "$S/" runs/shared/
PY=.venv/bin/python
$PY - <<'PYEOF'
import json
from pathlib import Path
root = Path("runs/shared")
for done in root.rglob("*.done"):
    meta = json.loads(done.read_text()); target = done.with_suffix("")
    if meta["split"] and not (target.exists() and target.stat().st_size == meta["bytes"]):
        parts = sorted(target.parent.glob(target.name + ".part*"))
        assert len(parts) == meta["parts"], (target, len(parts))
        with open(target, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
        for p in parts:
            p.unlink()
    assert target.stat().st_size == meta["bytes"], target
    print("ok", target.relative_to(root), meta["bytes"])
PYEOF
cp runs/shared/data/cache/folds_s1_k5.parquet data/cache/folds_s1_k5.parquet     # the GPU box's exact folds
$PY - <<'PYEOF'
import pandas as pd, pyarrow.parquet as pq
from pathlib import Path
files = sorted(Path("runs/shared/E015/train_chunks").glob("*.parquet"))
want = ["s1_id", "cand_id", "prob", "cos_name", "name_ratio", "name_jw", "name_full_tset", "name_tsort"]
cols = [c for c in want if c in pq.read_schema(files[0]).names]
df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
Path("runs/import/export").mkdir(parents=True, exist_ok=True)
df.to_parquet("runs/import/export/train_min.parquet", index=False)
print("train_min", df.shape, cols)
PYEOF
mkdir -p runs/import/export && ln -sfn "$PWD/runs/shared/frames/E019" runs/import/export/frames
ls -la runs/import/export/ runs/import/export/frames/ && echo FETCH OK
