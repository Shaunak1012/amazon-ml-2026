#!/usr/bin/env bash
# Stage owner-model inputs for the remote GPU box in s3://<bucket>/shreyas-gpu/data/ (the GPU box reads them through
# presigned links only). Test groups come from OW01's test scoring (same prep settings as OW02).
set -euo pipefail
D=s3://$S3_BUCKET/shreyas-gpu/data
until [ -f runs/OW01/infer_test_groups.parquet ]; do sleep 30; done
for f in train_groups infer_train_groups; do aws s3 cp --only-show-errors runs/OW02/$f.parquet $D/OW/$f.parquet; done
aws s3 cp --only-show-errors runs/OW02/prep.json $D/OW/prep.json
aws s3 cp --only-show-errors runs/OW01/infer_test_groups.parquet $D/OW/infer_test_groups.parquet
for s in train test; do for k in 1 2 3; do aws s3 cp --only-show-errors data/cache/${s}_s${k}.parquet $D/cache/${s}_s${k}.parquet; done; done
echo staged > /tmp/staged && aws s3 cp --only-show-errors /tmp/staged $D/STAGED
aws s3 ls $D/ --recursive --human-readable
