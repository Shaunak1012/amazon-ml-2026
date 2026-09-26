#!/usr/bin/env bash
# Build the SH01 submission once Shaunak's E020 test frame (+ his sub_E020 outputs, for the diff) is uploaded.
set -euo pipefail
S=s3://$S3_BUCKET/shared/runs
until aws s3 ls "$S/frames/E020/test_frame.parquet.done" >/dev/null 2>&1; do sleep 60; done
bash scripts/sm/pull_code.sh
aws s3 sync --only-show-errors --exclude "E015/train_chunks/*" "$S/" runs/shared/
PY=.venv/bin/python
$PY - <<'PYEOF'
import json
from pathlib import Path
for done in Path("runs/shared").rglob("*.done"):
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
PYEOF
PYTHONPATH=. $PY scripts/sm/jobs/sh01_submit.py --frame runs/shared/frames/E020/test_frame.parquet \
    --tags SH01both SH01e12 SH01e13 SH01f30d --out submissions/sub_SH01 --ref runs/shared/E015/sub_E020
aws s3 cp --only-show-errors "s3://$S3_BUCKET/$S3_PREFIX/code/validate_submission.py" data/validate_submission.py   # organisers'
python3 data/validate_submission.py --matching submissions/sub_SH01/matching_results.tsv --candidate submissions/sub_SH01/candidate_pairs.tsv \
    --test-dir data/dataset/test --check-ids
aws s3 cp --only-show-errors --recursive submissions/sub_SH01/ "s3://$S3_BUCKET/$S3_PREFIX/artifacts/sub_SH01/" --exclude "*.parquet"
echo SUBMIT FILES READY
