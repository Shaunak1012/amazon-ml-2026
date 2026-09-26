#!/usr/bin/env bash
# CE03 staging: the strict candidate set (stage-1 p >= 0.02 OR ce_score_2 >= 0.02) of the fold-0 train frame and the E020
# test frame, as [s1_id, cand_id] chunk files for GPU scoring; plus the train chunks + folds for CE training.
set -euo pipefail
bash scripts/sm/pull_code.sh
.venv/bin/python - <<'PYEOF'
import numpy as np
import pandas as pd
from pathlib import Path
for split, f in (("train", "runs/import/export/frames/train_frame.parquet"), ("test", "runs/shared/frames/E020/test_frame.parquet")):
    X = pd.read_parquet(f, columns=["s1_id", "cand_id", "prob", "ce_score_2"])
    keep = (X.prob.to_numpy() >= 0.02) | (np.nan_to_num(X.ce_score_2.to_numpy()) >= 0.02)
    P = X.loc[keep, ["s1_id", "cand_id"]].reset_index(drop=True)
    d = Path("runs/CE03") / f"pairs_{split}"
    d.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(range(0, len(P), 1_000_000)):
        P.iloc[s:s + 1_000_000].to_parquet(d / f"{i:03d}.parquet", index=False)
    print(f"{split}: {len(P):,} pairs, {len(P) / X.s1_id.nunique():.2f} per S1")
PYEOF
aws s3 sync --only-show-errors runs/CE03/ "s3://$S3_BUCKET/shreyas-gpu/data/CE03/"
echo staged | aws s3 cp - "s3://$S3_BUCKET/shreyas-gpu/data/CE03/STAGED"
aws s3 ls "s3://$S3_BUCKET/shreyas-gpu/data/CE03/" --recursive | wc -l
