#!/usr/bin/env bash
# One-time setup of the SageMaker JupyterLab space (CPU box, e.g. ml.r7i.8xlarge). Idempotent: safe to re-run.
# Run in the space's terminal:
#   aws s3 cp s3://amazon-sagemaker-548171706001-ap-southeast-2-bjmz86wfr8l6oy/shreyas/bootstrap.sh - | bash
set -euo pipefail
BUCKET=amazon-sagemaker-548171706001-ap-southeast-2-bjmz86wfr8l6oy
REPO=$HOME/amazon-ml-2026
cd "$HOME"

# 1. code: snapshot of the shreyas branch from S3 (no GitHub auth on this box; it never pushes)
mkdir -p "$REPO" && cd "$REPO"
aws s3 cp --only-show-errors "s3://$BUCKET/shreyas/code/repo.tar.gz" - | tar -xz -C "$REPO"
echo "code at $(cat COMMIT 2>/dev/null)"

# 2. python env (CPU only)
if [ ! -x "$REPO/.venv/bin/python" ]; then python3 -m venv "$REPO/.venv"; fi
"$REPO/.venv/bin/pip" install -q --upgrade pip
"$REPO/.venv/bin/pip" install -q -r requirements-core.txt boto3
# pipeline modules import torch/transformers at load time; CPU wheels are enough here
"$REPO/.venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$REPO/.venv/bin/pip" install -q transformers==5.17.0 tokenizers==0.23.2
echo "python: $("$REPO/.venv/bin/python" --version)"

# 3. data: organiser zip from the bucket -> data/dataset/{train,test}
if [ ! -f data/dataset/test/test_source1.tsv ]; then
  mkdir -p data/_zip && aws s3 cp --only-show-errors "s3://$BUCKET/6ab10eb3b23ba_student_resource.zip" data/_zip/sr.zip
  (cd data/_zip && python3 -m zipfile -e sr.zip .)
  src=$(dirname "$(find data/_zip -path '*dataset/test/test_source1.tsv' | grep -v __MACOSX | head -1)")/..
  mkdir -p data/dataset && cp -r "$src/train" "$src/test" data/dataset/ && rm -rf data/_zip
fi
wc -l data/dataset/test/test_source1.tsv
[ -f .env ] || printf 'DATA_DIR=./data\nRUNS_DIR=./runs\nOOF_DIR=./oof\n' > .env

# 4. runner (restart if already running)
pkill -f scripts/sm/runner.py || true
mkdir -p data/cache
nohup "$REPO/.venv/bin/python" scripts/sm/runner.py --bucket "$BUCKET" > "$HOME/runner.out" 2>&1 &
sleep 3 && echo "runner pid $(pgrep -f scripts/sm/runner.py)" && echo "BOOTSTRAP OK: you can close this terminal."
