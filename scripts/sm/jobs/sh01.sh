#!/usr/bin/env bash
# SH01 arm on the CPU box (set ARM=eval | both on the first line). Waits for the phase-1 import (E019 train frame).
set -euo pipefail
ARM=${ARM:?set ARM=eval or ARM=both}
until [ -f runs/import/export/frames/train_frame.parquet ] && [ -f runs/import/export/train_min.parquet ]; do sleep 60; done
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
$PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
    ${COMP:+--comp-cols $COMP} --drop-s1-frac 0.19 --drop-in "$ARM" --tag "SH01$ARM" --lgb-params '{"num_threads": 16}'
aws s3 cp --only-show-errors "runs/frames2/SH01$ARM/result.json" "s3://$S3_BUCKET/$S3_PREFIX/artifacts/SH01$ARM/result.json"
cat "runs/frames2/SH01$ARM/result.json"
