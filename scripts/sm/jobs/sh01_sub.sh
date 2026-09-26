#!/usr/bin/env bash
# SH01 third arm: current features, fit on the same reduced S1 set as 'both' (isolates the recompute effect); then
# score all arms under both decision rules.
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
$PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
    ${COMP:+--comp-cols $COMP} --drop-s1-frac 0.19 --drop-in eval --fit-without-dropped --tag SH01evalsub \
    --lgb-params '{"num_threads": 32}'
PYTHONPATH=. $PY scripts/sm/jobs/sh01_rules.py | tee runs/frames2/sh01_rules.json
aws s3 cp --only-show-errors runs/frames2/sh01_rules.json "s3://$S3_BUCKET/$S3_PREFIX/artifacts/sh01_rules.json"
