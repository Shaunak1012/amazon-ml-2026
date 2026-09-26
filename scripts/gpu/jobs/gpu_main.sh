#!/usr/bin/env bash
# GPU main job: OW03 (owner model, e5-base, 2 epochs) then CE03 (multilingual-e5-large cross-encoder, MIT 560M).
# Every step is guarded (skips finished work) and waits for any other er_owner/er_crossenc process first, so an
# overlapping earlier queue entry can never run the same training twice. Inputs via presigned links only.
set -euo pipefail
PY=.venv/bin/python
dl() {  # url dest: retry until the object exists
  for i in $(seq 1 120); do [ -s "$2" ] && return 0; curl -sf "$1" -o "$2.tmp" && mv "$2.tmp" "$2" && return 0; sleep 30; done
  echo "download failed: $2"; return 1; }
wait_gpu() { while pgrep -f "src.er_(owner|crossenc)" >/dev/null; do sleep 30; done; }
up() { $PY scripts/gpu/put.py "$1" "$2"; }
$PY -m pip install -q psutil nvidia-ml-py
mkdir -p runs/OW03 runs/CE03/pairs_train runs/CE03/pairs_test runs/E015/train_chunks data/cache
dl "{{GET:shreyas-gpu/data/cache/train_s1.parquet}}" data/cache/train_s1.parquet
dl "{{GET:shreyas-gpu/data/cache/train_s2.parquet}}" data/cache/train_s2.parquet
dl "{{GET:shreyas-gpu/data/cache/train_s3.parquet}}" data/cache/train_s3.parquet
dl "{{GET:shreyas-gpu/data/cache/test_s1.parquet}}" data/cache/test_s1.parquet
dl "{{GET:shreyas-gpu/data/cache/test_s2.parquet}}" data/cache/test_s2.parquet
dl "{{GET:shreyas-gpu/data/cache/test_s3.parquet}}" data/cache/test_s3.parquet
dl "{{GET:shared/runs/data/cache/folds_s1_k5.parquet}}" data/cache/folds_s1_k5.parquet
# ---------------------------------------------------------------- OW03: owner model
D=runs/OW03
dl "{{GET:shreyas-gpu/data/OW/train_groups.parquet}}" $D/train_groups.parquet
dl "{{GET:shreyas-gpu/data/OW/infer_train_groups.parquet}}" $D/infer_train_groups.parquet
dl "{{GET:shreyas-gpu/data/OW/infer_test_groups.parquet}}" $D/infer_test_groups.parquet
dl "{{GET:shreyas-gpu/data/OW/prep.json}}" $D/prep.json
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
wait_gpu
[ -f $D/owner_model.pt ] || $PY -m src.er_owner train --dir $D --model intfloat/multilingual-e5-base --epochs 2 \
    --batch 64 --lr 3e-5 --n 200000 --bf16 --ckpt-every 500
up $D/train.json artifacts/OW03/train.json
wait_gpu
[ -f $D/train_owner.parquet ] || $PY -m src.er_owner score --dir $D --split train --batch 256 --bf16
up $D/train_owner.parquet artifacts/OW03/train_owner.parquet
[ -f $D/test_owner.parquet ] || $PY -m src.er_owner score --dir $D --split test --batch 256 --bf16
up $D/test_owner.parquet artifacts/OW03/test_owner.parquet
echo "OW03 DONE $(date)"
# ---------------------------------------------------------------- CE03: e5-large cross-encoder
{{GETDIR:shared/runs/E015/train_chunks/:runs/E015/train_chunks}}
{{GETDIR:shreyas-gpu/data/CE03/pairs_train/:runs/CE03/pairs_train}}
{{GETDIR:shreyas-gpu/data/CE03/pairs_test/:runs/CE03/pairs_test}}
wait_gpu
$PY -m src.er_crossenc train --chunks runs/E015/train_chunks --out runs/CE03/model --n __CE_N__ --exclude-folds 0 \
    --model-name intfloat/multilingual-e5-large --batch 64 --lr 1.5e-5 --ckpt-every 2000
$PY -m src.er_crossenc score --model runs/CE03/model --chunks runs/CE03/pairs_train --split train --out runs/CE03/train_ce --batch 512
for f in runs/CE03/train_ce/*.parquet; do up "$f" "artifacts/CE03/train_ce/$(basename "$f")"; done
$PY -m src.er_crossenc score --model runs/CE03/model --chunks runs/CE03/pairs_test --split test --out runs/CE03/test_ce --batch 512
for f in runs/CE03/test_ce/*.parquet; do up "$f" "artifacts/CE03/test_ce/$(basename "$f")"; done
echo "CE03 DONE $(date)"
