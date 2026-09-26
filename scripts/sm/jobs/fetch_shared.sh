#!/usr/bin/env bash
# SageMaker box: wait for Shaunak's uploads (scripts/upload_to_shreyas.py -> s3://<bucket>/shared/runs/), download,
# reassemble split files, build train_min from the stage-1 train chunks, and lay out runs/import/export/ for SH01.
set -euo pipefail
S=s3://$S3_BUCKET/shared/runs
need=(frames/E019/train_frame.parquet frames/E019/frames.json data/cache/folds_s1_k5.parquet train_min.parquet)
for k in "${need[@]}"; do until aws s3 ls "$S/$k.done" >/dev/null 2>&1; do sleep 60; done; echo "ready: $k"; done
mkdir -p runs/shared && aws s3 sync --only-show-errors --exclude "E015/train_chunks/*" "$S/" runs/shared/
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
mkdir -p runs/import/export && cp runs/shared/train_min.parquet runs/import/export/train_min.parquet && ln -sfn "$PWD/runs/shared/frames/E019" runs/import/export/frames
ls -la runs/import/export/ runs/import/export/frames/ && echo FETCH OK
