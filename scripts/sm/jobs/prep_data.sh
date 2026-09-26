#!/usr/bin/env bash
# SageMaker CPU box: parquet cache + normalised names (rarity features need data/cache/*_norm.parquet).
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
$PY -m src.er_data
$PY -m src.er_normalize --workers 30
ls -la data/cache
