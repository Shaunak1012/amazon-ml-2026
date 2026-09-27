#!/usr/bin/env bash
# OW04: owner model on multilingual-e5-large (MIT, 560M), same groups as OW03 (already on this box), 2 epochs.
set -euo pipefail
PY=.venv/bin/python
D=runs/OW04
mkdir -p $D
cp -n runs/OW03/{train_groups,infer_train_groups,infer_test_groups}.parquet runs/OW03/prep.json $D/
while pgrep -f "src.er_(owner|crossenc)" >/dev/null; do sleep 30; done
[ -f $D/owner_model.pt ] || $PY -m src.er_owner train --dir $D --model intfloat/multilingual-e5-large --epochs 2 \
    --batch 32 --lr 2e-5 --n 200000 --bf16 --ckpt-every 1000
$PY scripts/gpu/put.py $D/train.json artifacts/OW04/train.json
[ -f $D/train_owner.parquet ] || $PY -m src.er_owner score --dir $D --split train --batch 256 --bf16
$PY scripts/gpu/put.py $D/train_owner.parquet artifacts/OW04/train_owner.parquet
[ -f $D/test_owner.parquet ] || $PY -m src.er_owner score --dir $D --split test --batch 256 --bf16
$PY scripts/gpu/put.py $D/test_owner.parquet artifacts/OW04/test_owner.parquet
echo "OW04 DONE $(date)"
