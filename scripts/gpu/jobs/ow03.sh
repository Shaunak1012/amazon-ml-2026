#!/usr/bin/env bash
# OW03: listwise owner model, multilingual-e5-base (MIT), 2 epochs on the remote GPU; scores fold-0 + test groups.
set -euo pipefail
PY=.venv/bin/python
D=runs/OW03
mkdir -p $D data/cache
dl() {  # url dest: retry until the object exists (staging may still be running)
  for i in $(seq 1 120); do [ -s "$2" ] && return 0; curl -sf "$1" -o "$2.tmp" && mv "$2.tmp" "$2" && return 0; sleep 30; done
  echo "download failed: $2"; return 1; }
dl "{{GET:shreyas-gpu/data/STAGED}}" /tmp/staged_ow03
dl "{{GET:shreyas-gpu/data/OW/train_groups.parquet}}" $D/train_groups.parquet
dl "{{GET:shreyas-gpu/data/OW/infer_train_groups.parquet}}" $D/infer_train_groups.parquet
dl "{{GET:shreyas-gpu/data/OW/infer_test_groups.parquet}}" $D/infer_test_groups.parquet
dl "{{GET:shreyas-gpu/data/OW/prep.json}}" $D/prep.json
dl "{{GET:shreyas-gpu/data/cache/train_s1.parquet}}" data/cache/train_s1.parquet
dl "{{GET:shreyas-gpu/data/cache/train_s2.parquet}}" data/cache/train_s2.parquet
dl "{{GET:shreyas-gpu/data/cache/train_s3.parquet}}" data/cache/train_s3.parquet
dl "{{GET:shreyas-gpu/data/cache/test_s1.parquet}}" data/cache/test_s1.parquet
dl "{{GET:shreyas-gpu/data/cache/test_s2.parquet}}" data/cache/test_s2.parquet
dl "{{GET:shreyas-gpu/data/cache/test_s3.parquet}}" data/cache/test_s3.parquet
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
[ -f $D/owner_model.pt ] || $PY -m src.er_owner train --dir $D --model intfloat/multilingual-e5-base --epochs 2 \
    --batch 64 --lr 3e-5 --n 200000 --bf16 --ckpt-every 500
$PY scripts/gpu/put.py $D/train.json artifacts/OW03/train.json
[ -f $D/train_owner.parquet ] || $PY -m src.er_owner score --dir $D --split train --batch 256 --bf16
$PY scripts/gpu/put.py $D/train_owner.parquet artifacts/OW03/train_owner.parquet
[ -f $D/test_owner.parquet ] || $PY -m src.er_owner score --dir $D --split test --batch 256 --bf16
$PY scripts/gpu/put.py $D/test_owner.parquet artifacts/OW03/test_owner.parquet
echo OW03 GPU DONE
