#!/usr/bin/env bash
# OW02: owner model with the slot-list leakage rule (OW01 kept only 48k groups) + length-bucketed batches.
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
D=runs/OW02
up() { aws s3 cp --only-show-errors "$1" "s3://$S3_BUCKET/$S3_PREFIX/artifacts/OW02/$(basename "$1")"; }
[ -f $D/train_groups.parquet ] || $PY -m src.er_owner prep --pairs runs/import/export/train_min.parquet \
    --frame runs/import/export/frames/train_frame.parquet --out $D
# train only once OW01's CPU-heavy steps are done (they share the 16 physical cores)
until aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/done/60_ow01_train.json" >/dev/null 2>&1; do sleep 60; done
[ -f $D/owner_model.pt ] || $PY -m src.er_owner train --dir $D --n 200000 --threads 16
up $D/train.json
[ -f $D/train_owner.parquet ] || $PY -m src.er_owner score --dir $D --split train --threads 16 --bf16
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
run() {  # tag, extra args
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" --lgb-params '{"num_threads": 16}' "${@:2}" > "runs/frames2_$1.log" 2>&1 \
      || echo "FAILED $1"
}
run OW02_k15 --extra-feats $D &
run OW02_or --extra-feats $D --cand-min-prob 0.02 --cand-ce-col ce_score_2 --cand-ce-min 0.02 &
wait
$PY - <<'PYEOF' | tee $D/compare.txt
import json
from pathlib import Path
def r(t):
    f = Path("runs/frames2") / t / "result.json"
    return json.loads(f.read_text()) if f.exists() else None
for base, new in (("CS_k15", "OW02_k15"), ("CS_or02c02", "OW02_or"), ("OW01_k15", "OW02_k15")):
    b, n = r(base), r(new)
    if b and n:
        print(f"{new}: dev {n['dev_f05']:.5f} vs {base} {b['dev_f05']:.5f} -> {n['dev_f05'] - b['dev_f05']:+.5f} | "
              f"US {n['dev_by_country']['US'] - b['dev_by_country']['US']:+.5f} India "
              f"{n['dev_by_country']['India'] - b['dev_by_country']['India']:+.5f} | cand/S1 {n['cand_per_s1_dev']:.2f}")
PYEOF
up $D/compare.txt
echo OW02 TRAIN-SIDE DONE
