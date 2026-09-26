#!/usr/bin/env bash
set -euo pipefail
bash scripts/sm/pull_code.sh
aws s3 sync --only-show-errors --exclude "*.done" "s3://$S3_BUCKET/shared/runs/E015/sub_E016/" runs/shared/E015/sub_E016/
PYTHONPATH=. .venv/bin/python scripts/sm/jobs/release_sizing.py runs/shared/E015/sub_E016 | tee runs/jd01_sizing.json
aws s3 cp --only-show-errors runs/jd01_sizing.json "s3://$S3_BUCKET/$S3_PREFIX/artifacts/jd01_sizing.json"
