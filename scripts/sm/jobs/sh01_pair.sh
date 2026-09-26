#!/usr/bin/env bash
# Paired baselines for SH01s12 / SH01s13: current recipe scored on the same dev subsets (drop seeds 12, 13).
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
for s in 12 13; do
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --drop-s1-frac 0.19 --drop-seed $s --drop-in eval --tag SH01eval_s$s \
      --lgb-params '{"num_threads": 16}' &
done
wait
PYTHONPATH=. $PY scripts/sm/jobs/sh01_rules.py | tee runs/frames2/sh01_rules.json
aws s3 cp --only-show-errors runs/frames2/sh01_rules.json "s3://$S3_BUCKET/$S3_PREFIX/artifacts/sh01_rules_pair.json"
