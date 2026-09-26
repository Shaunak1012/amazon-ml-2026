#!/usr/bin/env bash
# CS01: candidate-set size vs dev F0.5. Prune the stage-1 top-15 by rank and/or stage-1 prob, recompute competition
# features on the pruned population, refit stage 2 (normal density), report F0.5 + candidates/S1 + recall.
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
run() {  # tag topk minprob
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --cand-topk "$2" --cand-min-prob "$3" --tag "$1" --lgb-params '{"num_threads": 8}' \
      > "runs/frames2_$1.log" 2>&1 || echo "FAILED $1"
}
export -f run; export PY COMP
printf '%s\n' "CS_k15 15 0" "CS_k10 10 0" "CS_k8 8 0" "CS_k6 6 0" "CS_k5 5 0" "CS_p01 15 0.01" "CS_p02 15 0.02" \
  "CS_p05 15 0.05" "CS_k8p01 8 0.01" "CS_k6p02 6 0.02" | xargs -P 4 -L 1 bash -c 'run $0 $1 $2'
$PY - <<'PYEOF' | tee runs/frames2/cand_sweep.tsv
import json
from pathlib import Path
print("tag\ttopk\tminprob\tcand_per_s1\trecall\tno_cands\tdev_f05\tUS\tIndia\trule")
for d in sorted(Path("runs/frames2").glob("CS_*")):
    f = d / "result.json"
    if f.exists():
        r = json.loads(f.read_text())
        print(f"{r['tag']}\t{r['cand_topk']}\t{r['cand_min_prob']}\t{r['cand_per_s1_dev']:.2f}\t{r['cand_recall_dev']:.5f}"
              f"\t{r['s1_with_no_cands_dev']:.4f}\t{r['dev_f05']:.5f}\t{r['dev_by_country']['US']:.5f}"
              f"\t{r['dev_by_country']['India']:.5f}\t{r['stage2_rule']}")
PYEOF
aws s3 cp --only-show-errors runs/frames2/cand_sweep.tsv "s3://$S3_BUCKET/$S3_PREFIX/artifacts/cand_sweep.tsv"
