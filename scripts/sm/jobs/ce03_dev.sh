#!/usr/bin/env bash
# CE03 dev-only comparison (test scores still running on the GPU): clean filter, CE03 alone and CE03 + OW03.
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
O=s3://$S3_BUCKET/shreyas-gpu/out/artifacts/CE03
mkdir -p runs/CE03dev
aws s3 sync --only-show-errors --exclude "*.done" "$O/train_ce/" runs/CE03dev/train_ce/
$PY -c "
import pandas as pd; from pathlib import Path
f=sorted(Path('runs/CE03dev/train_ce').glob('*.parquet'))
d=pd.concat([pd.read_parquet(x) for x in f]).rename(columns={'ce_score':'ce3'}); d.to_parquet('runs/CE03dev/train_feats.parquet', index=False)
print(len(d), 'pairs, mean ce3', round(float(d.ce3.mean()),4))"
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" --cand-min-prob 0.02 --cand-ce-col ce_score --cand-ce-min 0.02 \
      --lgb-params '{"num_threads": 16}' "${@:2}" > "runs/frames2_$1.log" 2>&1 || { echo "FAILED $1"; tail -8 "runs/frames2_$1.log"; }
}
run CE03d_or --extra-feats runs/CE03dev &
run CE03dOW_or --extra-feats runs/CE03dev runs/OW03 &
wait
$PY - <<'PYEOF' | tee runs/CE03dev/compare.txt
import json
from pathlib import Path
r = lambda t: json.loads((Path("runs/frames2") / t / "result.json").read_text())
for t in ("CF_a02b02", "OW03_or", "CE03d_or", "CE03dOW_or"):
    x = r(t)
    print(f"{t}: dev {x['dev_f05']:.5f} US {x['dev_by_country']['US']:.5f} India {x['dev_by_country']['India']:.5f} cand/S1 {x['cand_per_s1_dev']:.2f}")
PYEOF
aws s3 cp --only-show-errors runs/CE03dev/compare.txt "s3://$S3_BUCKET/$S3_PREFIX/artifacts/CE03/compare_dev.txt"
