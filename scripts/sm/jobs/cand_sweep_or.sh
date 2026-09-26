#!/usr/bin/env bash
# CS04: strict OR-rule candidate sets (stage-1 prob >= p OR E016 CE (ce_score_2) >= c); every stage-2 population
# feature is recomputed on the pruned set, so candidate_pairs.tsv = everything the final model's features touch.
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
run() {  # tag minprob ce_min
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --cand-min-prob "$2" --cand-ce-col ce_score_2 --cand-ce-min "$3" --tag "$1" \
      --lgb-params '{"num_threads": 8}' > "runs/frames2_$1.log" 2>&1 || echo "FAILED $1"
}
export -f run; export PY COMP
for v in "CS_or02c01 0.02 0.01" "CS_or01c01 0.01 0.01" "CS_or02c02 0.02 0.02" "CS_or05c01 0.05 0.01"; do echo "$v"; done \
  | xargs -P 4 -L 1 bash -c 'run $0 $1 $2'
$PY - <<'PYEOF' | tee runs/frames2/cand_sweep_or.tsv
import json
from pathlib import Path
print("tag\tminprob\tcand_per_s1\trecall\tno_cands\tdev_f05\tUS\tIndia")
for d in sorted(Path("runs/frames2").glob("CS_*")):
    f = d / "result.json"
    if f.exists():
        r = json.loads(f.read_text())
        print(f"{r['tag']}\t{r['cand_min_prob']}\t{r['cand_per_s1_dev']:.2f}\t{r['cand_recall_dev']:.5f}"
              f"\t{r['s1_with_no_cands_dev']:.4f}\t{r['dev_f05']:.5f}\t{r['dev_by_country']['US']:.5f}\t{r['dev_by_country']['India']:.5f}")
PYEOF
aws s3 cp --only-show-errors runs/frames2/cand_sweep_or.tsv "s3://$S3_BUCKET/$S3_PREFIX/artifacts/cand_sweep_or.tsv"
