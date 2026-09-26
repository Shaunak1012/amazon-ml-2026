#!/usr/bin/env bash
# SH01 ensemble members: training drop seed/fraction varies, dev population fixed (seed 11, 19%) = same as SH01both.
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
for v in "SH01e12 0.19 12" "SH01e13 0.19 13" "SH01f30d 0.30 14"; do
  set -- $v
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --drop-s1-frac "$2" --drop-seed "$3" --dev-drop-seed 11 --dev-drop-frac 0.19 \
      --drop-in both --tag "$1" --lgb-params '{"num_threads": 16}' &
done
wait
PYTHONPATH=. $PY scripts/sm/jobs/sh01_rules.py | tee runs/frames2/sh01_rules.json
aws s3 cp --only-show-errors runs/frames2/sh01_rules.json "s3://$S3_BUCKET/$S3_PREFIX/artifacts/sh01_rules_ens.json"
