#!/usr/bin/env bash
# Probe: can this space's role launch a Spot ml.g5.2xlarge training job (permissions, quota, capacity)?
# The SageMaker SDK gets its own venv so it can never change the pipeline venv's pinned numpy/pandas.
set -euo pipefail
bash scripts/sm/pull_code.sh
[ -x .venv_sm/bin/python ] || python3 -m venv .venv_sm
.venv_sm/bin/pip install -q "sagemaker>=2.200,<3"
.venv_sm/bin/python scripts/sm/gpu/launch.py --entry probe.py --name probe --max-run 900 --max-wait 3600 --bucket "$S3_BUCKET"
