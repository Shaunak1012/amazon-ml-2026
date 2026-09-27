#!/usr/bin/env bash
# E23-regime paired comparisons (Shaunak's E023b settings, logic copied into er_frames2): comp-keep 0.78 density
# (fit + dev always kept, seed 11), candidate rule applied AFTER features: stage-1 p >= 0.2 OR E016 CE (ce_score_2 on
# the train side) >= 0.01. Identical dev rows in every arm. (a) base (b) +OW03 (c) +CE03 (d) +OW03+CE03,
# then (e) +OW04 (f) +OW04+CE03 once OW04's fold-0 scores are in S3.
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
REG="--comp-keep 0.78 --prune-post --cand-min-prob 0.2 --cand-ce-col ce_score_2 --cand-ce-min 0.01"
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" $REG --lgb-params '{"num_threads": 8}' "${@:2}" > "runs/frames2_$1.log" 2>&1 \
      || { echo "FAILED $1"; tail -8 "runs/frames2_$1.log"; }
}
report() {
$PY - "$@" <<'PYEOF' | tee runs/frames2/e23_paired.txt
import json, sys
from pathlib import Path
rows = []
for t in ("E23_base", "E23_ow03", "E23_ce03", "E23_ow03ce03", "E23_ow04", "E23_ow04ce03"):
    f = Path("runs/frames2") / t / "result.json"
    if f.exists():
        rows.append((t, json.loads(f.read_text())))
b = dict(rows).get("E23_base")
print("arm\tdev_f05\tdelta\tUS\tIndia\tcand_per_s1\trecall\tn_dev\trule")
for t, r in rows:
    d = r["dev_f05"] - b["dev_f05"] if b else float("nan")
    print(f"{t}\t{r['dev_f05']:.5f}\t{d:+.5f}\t{r['dev_by_country']['US']:.5f}\t{r['dev_by_country']['India']:.5f}"
          f"\t{r['cand_per_s1_dev']:.3f}\t{r['cand_recall_dev']:.5f}\t{r['n_dev']}\t{r['stage2_rule']}")
PYEOF
aws s3 cp --only-show-errors runs/frames2/e23_paired.txt "s3://$S3_BUCKET/$S3_PREFIX/artifacts/e23_paired.txt"
}
run E23_base &
run E23_ow03 --extra-feats runs/OW03 &
run E23_ce03 --extra-feats runs/CE03dev &
run E23_ow03ce03 --extra-feats runs/CE03dev runs/OW03 &
wait
report
# OW04: same regime, same rows, once its fold-0 (train-side) scores exist
O=s3://$S3_BUCKET/shreyas-gpu/out/artifacts/OW04
until aws s3 ls "$O/train_owner.parquet.done" >/dev/null 2>&1; do sleep 60; done
mkdir -p runs/OW04dev
n=$(aws s3 cp "$O/train_owner.parquet.done" - | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['parts'] if d['split'] else 0)")
if [ "$n" = 0 ]; then aws s3 cp --only-show-errors "$O/train_owner.parquet" runs/OW04dev/train_owner.parquet
else for i in $(seq 0 $((n - 1))); do aws s3 cp --only-show-errors "$O/train_owner.parquet.part$(printf %03d $i)" -; done > runs/OW04dev/train_owner.parquet; fi
run E23_ow04 --extra-feats runs/OW04dev &
run E23_ow04ce03 --extra-feats runs/CE03dev runs/OW04dev &
wait
report
echo E23 PAIRED DONE
