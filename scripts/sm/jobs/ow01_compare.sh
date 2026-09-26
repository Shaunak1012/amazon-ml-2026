#!/usr/bin/env bash
# Re-run OW01's paired stage-2 comparison (the first attempt globbed train_groups.parquet by mistake).
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
D=runs/OW01
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" --lgb-params '{"num_threads": 12}' "${@:2}" > "runs/frames2_$1.log" 2>&1 \
      || { echo "FAILED $1"; tail -15 "runs/frames2_$1.log"; }
}
run OW01_k15 --extra-feats $D &
run OW01_or --extra-feats $D --cand-min-prob 0.02 --cand-ce-col ce_score_2 --cand-ce-min 0.02 &
wait
$PY - <<'PYEOF' | tee $D/compare.txt
import json
from pathlib import Path
def r(t):
    f = Path("runs/frames2") / t / "result.json"
    return json.loads(f.read_text()) if f.exists() else None
for base, new in (("CS_k15", "OW01_k15"), ("CS_or02c02", "OW01_or")):
    b, n = r(base), r(new)
    if b and n:
        print(f"{new}: dev {n['dev_f05']:.5f} vs {base} {b['dev_f05']:.5f} -> {n['dev_f05'] - b['dev_f05']:+.5f} | "
              f"US {n['dev_by_country']['US'] - b['dev_by_country']['US']:+.5f} India "
              f"{n['dev_by_country']['India'] - b['dev_by_country']['India']:+.5f} | cand/S1 {n['cand_per_s1_dev']:.2f} | rule {n['stage2_rule']}")
PYEOF
aws s3 cp --only-show-errors $D/compare.txt "s3://$S3_BUCKET/$S3_PREFIX/artifacts/OW01/compare.txt"
