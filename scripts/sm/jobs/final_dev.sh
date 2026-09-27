#!/usr/bin/env bash
# FINAL config on dev: OW03 + CE03 features, CE03 cascade filter (p >= 0.05 OR ce3 >= 0.10), NO self-trained column
# (ce_score_2 dropped; team decision). A = normal training; B = test-density training. A_eval/B on the same test-like dev.
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
BASE="--cand-min-prob 0.05 --cand-ce-col ce3 --cand-ce-min 0.10 --extra-feats runs/CE03dev runs/OW03 --drop-cols ce_score_2"
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" $BASE --lgb-params '{"num_threads": 10}' "${@:2}" > "runs/frames2_$1.log" 2>&1 \
      || { echo "FAILED $1"; tail -8 "runs/frames2_$1.log"; }
}
run FA_n &
run FA_e --drop-s1-frac 0.19 --drop-in eval &
run FB --drop-s1-frac 0.19 --drop-in both &
wait
$PY - <<'PYEOF' | tee runs/frames2/final_dev.txt
import json
from pathlib import Path
for t in ("CF3_a05b10", "FA_n", "FA_e", "FB"):
    f = Path("runs/frames2") / t / "result.json"
    if f.exists():
        r = json.loads(f.read_text())
        print(f"{t}: dev {r['dev_f05']:.5f} US {r['dev_by_country']['US']:.5f} India {r['dev_by_country']['India']:.5f} "
              f"cand/S1 {r['cand_per_s1_dev']:.2f} recall {r['cand_recall_dev']:.5f} rule {r['stage2_rule']}")
PYEOF
aws s3 cp --only-show-errors runs/frames2/final_dev.txt "s3://$S3_BUCKET/$S3_PREFIX/artifacts/final_dev.txt"
