#!/usr/bin/env bash
# SageMaker CPU box: wait for the GPU box's export (scripts/export_for_shreyas.py), then unpack to runs/import/.
set -euo pipefail
B=s3://$S3_BUCKET/$S3_PREFIX/import
until aws s3 ls "$B/DONE.json" >/dev/null 2>&1; do sleep 60; done
N=$(aws s3 cp "$B/DONE.json" - | python3 -c "import json,sys; print(json.load(sys.stdin)['parts'])")
mkdir -p runs/import && rm -rf runs/import/export
for i in $(seq 0 $((N - 1))); do aws s3 cp --only-show-errors "$B/part-$(printf %02d "$i")" - ; done | tar -x -C runs/import
cp runs/import/export/folds_s1_k5.parquet data/cache/folds_s1_k5.parquet   # the GPU box's exact folds
du -sh runs/import/export/* && cat runs/import/export/frames/frames.json && echo IMPORT OK
