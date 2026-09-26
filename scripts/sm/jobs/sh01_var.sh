#!/usr/bin/env bash
# SH01 variant (first line sets TAG, FRAC, SEED): test-like stage 2 with another drop subset / fraction.
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
$PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
    ${COMP:+--comp-cols $COMP} --drop-s1-frac "$FRAC" --drop-seed "$SEED" --drop-in both --tag "$TAG" \
    --lgb-params '{"num_threads": 10}'
aws s3 cp --only-show-errors "runs/frames2/$TAG/result.json" "s3://$S3_BUCKET/$S3_PREFIX/artifacts/$TAG/result.json"
