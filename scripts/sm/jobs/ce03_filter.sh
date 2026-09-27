#!/usr/bin/env bash
# CF02: candidate filter using the e5-large CE (ce3) as a cascade stage: keep p >= a OR ce3 >= b (from the clean superset).
# Features OW03 + CE03, target <= 4.7 cand/S1. Prints cand/S1 on dev AND the rule's size on the test superset (for ce3 rows scored).
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
# merge ce3 into the train frame as a column the filter can use: frames2 filters on --cand-ce-col after add_extra
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" --cand-min-prob "$2" --cand-ce-col ce3 --cand-ce-min "$3" \
      --extra-feats runs/CE03dev runs/OW03 --lgb-params '{"num_threads": 8}' > "runs/frames2_$1.log" 2>&1 \
      || { echo "FAILED $1"; tail -8 "runs/frames2_$1.log"; }
}
export -f run; export PY COMP
for v in "CF3_a05b10 0.05 0.10" "CF3_a10b10 0.10 0.10" "CF3_a10b20 0.10 0.20" "CF3_a20b10 0.20 0.10"; do echo "$v"; done \
  | xargs -P 4 -L 1 bash -c 'run $0 $1 $2'
$PY - <<'PYEOF' | tee runs/frames2/ce03_filter.tsv
import json
from pathlib import Path
print("tag\tcand_per_s1\trecall\tdev_f05\tUS\tIndia")
for t in ("CE03dOW_or", "CF3_a05b10", "CF3_a10b10", "CF3_a10b20", "CF3_a20b10"):
    f = Path("runs/frames2") / t / "result.json"
    if f.exists():
        r = json.loads(f.read_text())
        print(f"{t}\t{r['cand_per_s1_dev']:.2f}\t{r['cand_recall_dev']:.5f}\t{r['dev_f05']:.5f}\t{r['dev_by_country']['US']:.5f}\t{r['dev_by_country']['India']:.5f}")
PYEOF
aws s3 cp --only-show-errors runs/frames2/ce03_filter.tsv "s3://$S3_BUCKET/$S3_PREFIX/artifacts/ce03_filter.tsv"
