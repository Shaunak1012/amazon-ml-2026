#!/usr/bin/env bash
# OW03 + clean filter, 3 LightGBM seeds -> averaged stage 2 (normal-density dev, comparable to OW03_or 0.99122).
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" --extra-feats runs/OW03 --cand-min-prob 0.02 --cand-ce-col ce_score \
      --cand-ce-min 0.02 --lgb-params "{\"num_threads\": 10, \"seed\": $2, \"bagging_seed\": $2, \"feature_fraction_seed\": $2}" \
      > "runs/frames2_$1.log" 2>&1 || { echo "FAILED $1"; tail -8 "runs/frames2_$1.log"; }
}
run OW03_or_s1 1 & run OW03_or_s2 2 & run OW03_or_s3 3 &
wait
PYTHONPATH=. $PY scripts/sm/jobs/ens_rules.py OW03_or OW03_or_s1 OW03_or_s2 OW03_or_s3 | tee runs/frames2/ow03_seeds.json
aws s3 cp --only-show-errors runs/frames2/ow03_seeds.json "s3://$S3_BUCKET/$S3_PREFIX/artifacts/ow03_seeds.json"
