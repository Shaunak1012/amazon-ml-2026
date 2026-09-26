#!/usr/bin/env bash
# CS03: on the real TEST candidates (E020 test frame), candidates/S1 by country at each stage-1 floor, and the share
# of stage-2-accepted pairs (Shaunak's E020 test probs) each floor would have cut. Flags a France-specific loss.
set -euo pipefail
S=s3://$S3_BUCKET/shared/runs
until aws s3 ls "$S/frames/E020/test_frame.parquet.done" >/dev/null 2>&1; do sleep 60; done
until [ -f runs/shared/frames/E020/test_frame.parquet ] && [ ! -f runs/shared/frames/E020/test_frame.parquet.part000 ]; do
  aws s3 sync --only-show-errors --exclude "E015/train_chunks/*" "$S/" runs/shared/; sleep 30; done
bash scripts/sm/pull_code.sh
.venv/bin/python - <<'PYEOF' | tee runs/cand_test_by_country.txt
import pandas as pd
from pathlib import Path
from src.er_data import cache_dir
X = pd.read_parquet("runs/shared/frames/E020/test_frame.parquet", columns=["s1_id", "cand_id", "prob"])
ctry = pd.read_parquet(cache_dir() / "test_s1_norm.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
X["country"] = ctry.reindex(X.s1_id).to_numpy()
n_s1 = ctry.value_counts()
ref = Path("runs/shared/E015/sub_E020/test_probs_stage2.parquet")
acc = None
if ref.exists():
    P = pd.read_parquet(ref)
    acc = X.merge(P.rename(columns={"prob": "p2"}), on=["s1_id", "cand_id"], how="left").p2.fillna(0).to_numpy() >= 0.5
print("floor\tcountry\tcand_per_s1\taccepted_pairs_cut_%")
for f in (0.0, 0.003, 0.005, 0.0075, 0.01, 0.02):
    keep = X.prob.to_numpy() >= f
    for c in n_s1.index:
        m = X.country.to_numpy() == c
        cut = "" if acc is None else f"{100 * (acc & m & ~keep).sum() / max((acc & m).sum(), 1):.3f}"
        print(f"{f}\t{c}\t{(keep & m).sum() / n_s1[c]:.2f}\t{cut}")
PYEOF
aws s3 cp --only-show-errors runs/cand_test_by_country.txt "s3://$S3_BUCKET/$S3_PREFIX/artifacts/cand_test_by_country.txt"
