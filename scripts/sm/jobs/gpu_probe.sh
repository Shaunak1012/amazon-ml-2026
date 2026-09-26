#!/usr/bin/env bash
# Probe: can this space's role launch a Spot ml.g5.2xlarge training job (permissions, quota, capacity)?
set -euo pipefail
bash scripts/sm/pull_code.sh
.venv/bin/pip install -q "sagemaker>=2.200,<3"
.venv/bin/python scripts/sm/gpu/launch.py --entry probe.py --name probe --max-run 900 --max-wait 3600 --bucket "$S3_BUCKET"
