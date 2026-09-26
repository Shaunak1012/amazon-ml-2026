#!/usr/bin/env bash
# SageMaker CPU box: wait for one export phase from the GPU box (scripts/export_for_shreyas.py --phase P), unpack it
# into runs/import/export/. Queue as a copy with PHASE set on the first line (train | test).
set -euo pipefail
PHASE=${PHASE:?set PHASE=train or PHASE=test}
B=s3://$S3_BUCKET/$S3_PREFIX/import/$PHASE
until aws s3 ls "$B/DONE.json" >/dev/null 2>&1; do sleep 60; done
N=$(aws s3 cp "$B/DONE.json" - | python3 -c "import json,sys; print(json.load(sys.stdin)['parts'])")
mkdir -p runs/import
for i in $(seq 0 $((N - 1))); do aws s3 cp --only-show-errors "$B/part-$(printf %02d "$i")" - ; done | tar -x -C runs/import
if [ "$PHASE" = train ]; then cp runs/import/export/folds_s1_k5.parquet data/cache/folds_s1_k5.parquet; fi   # exact folds
du -sh runs/import/export/* && echo "IMPORT $PHASE OK"
