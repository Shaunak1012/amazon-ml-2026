#!/usr/bin/env bash
# SageMaker box: refresh code from the S3 snapshot (published locally by scripts/sm/push_code.sh). Keeps data/runs/.venv.
set -euo pipefail
aws s3 cp --only-show-errors "s3://${S3_BUCKET}/${S3_PREFIX:-shreyas}/code/repo.tar.gz" - | tar -xz
echo "code at $(cat COMMIT)"
