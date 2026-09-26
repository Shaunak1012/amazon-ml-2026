#!/usr/bin/env bash
set -euo pipefail
bash scripts/sm/pull_code.sh
PYTHONPATH=. .venv/bin/python scripts/sm/jobs/owner_bench.py 2>&1 | grep -v Warning | tee runs/owner_bench.txt
aws s3 cp --only-show-errors runs/owner_bench.txt "s3://$S3_BUCKET/$S3_PREFIX/artifacts/owner_bench.txt"
