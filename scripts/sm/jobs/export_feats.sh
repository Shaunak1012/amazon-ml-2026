#!/usr/bin/env bash
# Hand-off: OW04 owner features + CE03 e5-large CE scores as single parquets keyed by (s1_id, cand_id), fold-0 + test.
set -euo pipefail
PY=.venv/bin/python
mkdir -p runs/handoff
$PY - <<'PYEOF'
import pandas as pd
from pathlib import Path
out = Path("runs/handoff")
for s in ("train", "test"):
    ce = pd.read_parquet(f"runs/CE03dev/{s}_feats.parquet")                     # s1_id, cand_id, ce3
    ow = pd.read_parquet(f"runs/OW04{'dev' if s == 'train' else ''}/{s}_owner.parquet" if s == "train" else "runs/OW04/test_owner.parquet")
    ce.to_parquet(out / f"ce3_{s}.parquet", index=False)
    ow.rename(columns=lambda c: c.replace("own_", "ow4_")).to_parquet(out / f"ow4_{s}.parquet", index=False)
    print(s, "ce3", ce.shape, "ow4", ow.shape)
PYEOF
aws s3 sync --only-show-errors runs/handoff/ "s3://$S3_BUCKET/$S3_PREFIX/artifacts/handoff/"
aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/artifacts/handoff/" --human-readable
