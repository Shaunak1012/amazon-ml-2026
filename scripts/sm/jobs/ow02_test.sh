#!/usr/bin/env bash
# OW02 test scoring: waits for the trained owner model and the fully reassembled E020 test frame.
set -euo pipefail
S=s3://$S3_BUCKET/shared/runs
until [ -f runs/OW02/owner_model.pt ]; do sleep 60; done
until aws s3 ls "$S/frames/E020/test_frame.parquet.done" >/dev/null 2>&1; do sleep 60; done
until [ -f runs/shared/frames/E020/test_frame.parquet ] && [ ! -e runs/shared/frames/E020/test_frame.parquet.part000 ]; do
  aws s3 sync --only-show-errors --exclude "E015/train_chunks/*" "$S/" runs/shared/
  .venv/bin/python - <<'PYEOF'
import json
from pathlib import Path
for done in Path("runs/shared").rglob("*.done"):
    meta = json.loads(done.read_text()); t = done.with_suffix("")
    parts = sorted(t.parent.glob(t.name + ".part*"))
    if meta["split"] and len(parts) == meta["parts"] and not (t.exists() and t.stat().st_size == meta["bytes"]):
        with open(t, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
        if t.stat().st_size == meta["bytes"]:
            for p in parts:
                p.unlink()
PYEOF
  sleep 20
done
bash scripts/sm/pull_code.sh
.venv/bin/python -m src.er_owner score --dir runs/OW02 --split test --threads 16 --bf16 \
    --test-frame runs/shared/frames/E020/test_frame.parquet
echo OW02 TEST DONE
