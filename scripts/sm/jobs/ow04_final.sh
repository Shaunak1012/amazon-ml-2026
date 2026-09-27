#!/usr/bin/env bash
# OW04 (e5-large owner model): paired dev vs OW03 in the final config (normal dev FA, test-like dev FB), then a v2 final
# build with OW04 (kept only if the dev comparison favours it; decided by a human after reading final_v2_compare.txt).
set -euo pipefail
O=s3://$S3_BUCKET/shreyas-gpu/out/artifacts/OW04
until aws s3 ls "$O/test_owner.parquet.done" >/dev/null 2>&1; do sleep 60; done
until [ -f runs/frames2/FINAL/predict.json ]; do sleep 60; done          # v1 final first
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
mkdir -p runs/OW04
for f in train_owner test_owner; do
  n=$(aws s3 cp "$O/$f.parquet.done" - | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['parts'] if d['split'] else 0)")
  if [ "$n" = 0 ]; then aws s3 cp --only-show-errors "$O/$f.parquet" runs/OW04/$f.parquet
  else for i in $(seq 0 $((n - 1))); do aws s3 cp --only-show-errors "$O/$f.parquet.part$(printf %03d $i)" -; done > runs/OW04/$f.parquet; fi
done
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
BASE="--cand-min-prob 0.05 --cand-ce-col ce3 --cand-ce-min 0.10 --extra-feats runs/CE03dev runs/OW04 --drop-cols ce_score_2"
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" $BASE --lgb-params '{"num_threads": 15}' "${@:2}" > "runs/frames2_$1.log" 2>&1 \
      || { echo "FAILED $1"; tail -8 "runs/frames2_$1.log"; }
}
run FA4_n &
run FB4 --drop-s1-frac 0.19 --drop-in both &
wait
$PY - <<'PYEOF' | tee runs/frames2/final_v2_compare.txt
import json
from pathlib import Path
r = lambda t: json.loads((Path("runs/frames2") / t / "result.json").read_text())
for a, b in (("FA_n", "FA4_n"), ("FB", "FB4")):
    x, y = r(a), r(b)
    print(f"{b} vs {a}: {y['dev_f05']:.5f} vs {x['dev_f05']:.5f} -> {y['dev_f05'] - x['dev_f05']:+.5f} | US "
          f"{y['dev_by_country']['US'] - x['dev_by_country']['US']:+.5f} India {y['dev_by_country']['India'] - x['dev_by_country']['India']:+.5f}")
PYEOF
aws s3 cp --only-show-errors runs/frames2/final_v2_compare.txt "s3://$S3_BUCKET/$S3_PREFIX/artifacts/final_v2_compare.txt"
OUT=submissions/final_shreyas_v2
$PY -m src.er_frames2 --frames runs/import/export/frames --test-frame runs/shared/frames/E020/test_frame.parquet \
    --train-min runs/import/export/train_min.parquet ${COMP:+--comp-cols $COMP} --tag FINALv2 $BASE \
    --drop-s1-frac 0.19 --drop-in both --lgb-params '{"num_threads": 30}' --out $OUT
python3 data/validate_submission.py --matching $OUT/matching_results.tsv --candidate $OUT/candidate_pairs.tsv \
    --test-dir data/dataset/test --check-ids | tail -3
for f in matching_results.tsv candidate_pairs.tsv; do aws s3 cp --only-show-errors $OUT/$f "s3://$S3_BUCKET/$S3_PREFIX/artifacts/final_shreyas_v2/$f"; done
echo FINAL V2 READY
