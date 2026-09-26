#!/usr/bin/env bash
# OW03 (GPU owner model, e5-base): fetch its features from shreyas-gpu/out, then the paired stage-2 dev comparison.
set -euo pipefail
O=s3://$S3_BUCKET/shreyas-gpu/out/artifacts/OW03
until aws s3 ls "$O/test_owner.parquet.done" >/dev/null 2>&1; do sleep 60; done
bash scripts/sm/pull_code.sh
mkdir -p runs/OW03
for f in train_owner test_owner; do
  n=$(aws s3 cp "$O/$f.parquet.done" - | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['parts'] if d['split'] else 0)")
  if [ "$n" = 0 ]; then aws s3 cp --only-show-errors "$O/$f.parquet" runs/OW03/$f.parquet
  else for i in $(seq 0 $((n - 1))); do aws s3 cp --only-show-errors "$O/$f.parquet.part$(printf %03d $i)" -; done > runs/OW03/$f.parquet; fi
done
aws s3 cp --only-show-errors "$O/train.json" runs/OW03/train.json
PY=.venv/bin/python
D=runs/OW03
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" --lgb-params '{"num_threads": 12}' "${@:2}" > "runs/frames2_$1.log" 2>&1 \
      || { echo "FAILED $1"; tail -15 "runs/frames2_$1.log"; }
}
run OW03_k15 --extra-feats $D &
run OW03_or --extra-feats $D --cand-min-prob 0.02 --cand-ce-col ce_score_2 --cand-ce-min 0.02 &
wait
$PY - <<'PYEOF' | tee $D/compare.txt
import json
from pathlib import Path
def r(t):
    f = Path("runs/frames2") / t / "result.json"
    return json.loads(f.read_text()) if f.exists() else None
for base, new in (("CS_k15", "OW03_k15"), ("CS_or02c02", "OW03_or"), ("OW01_k15", "OW03_k15")):
    b, n = r(base), r(new)
    if b and n:
        print(f"{new}: dev {n['dev_f05']:.5f} vs {base} {b['dev_f05']:.5f} -> {n['dev_f05'] - b['dev_f05']:+.5f} | "
              f"US {n['dev_by_country']['US'] - b['dev_by_country']['US']:+.5f} India "
              f"{n['dev_by_country']['India'] - b['dev_by_country']['India']:+.5f} | cand/S1 {n['cand_per_s1_dev']:.2f} | rule {n['stage2_rule']}")
PYEOF
aws s3 cp --only-show-errors $D/compare.txt "s3://$S3_BUCKET/$S3_PREFIX/artifacts/OW03/compare.txt"
