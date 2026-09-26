#!/usr/bin/env bash
# CE03 (GPU e5-large cross-encoder): collect its scores from shreyas-gpu/out, then paired stage-2 dev comparisons in the
# strict candidate regime (p >= 0.02 OR ce_score >= 0.02; clean E008 CE): CE03 alone, and CE03 + OW03 owner features.
set -euo pipefail
O=s3://$S3_BUCKET/shreyas-gpu/out/artifacts/CE03
until aws s3 ls "$O/test_ce/008.parquet.done" >/dev/null 2>&1; do sleep 60; done
until [ -f runs/frames2/CF_a02b02/result.json ]; do sleep 60; done
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
mkdir -p runs/CE03
aws s3 sync --only-show-errors --exclude "*.done" "$O/" runs/CE03/
$PY - <<'PYEOF'
import pandas as pd
from pathlib import Path
for s in ("train", "test"):
    f = sorted(Path(f"runs/CE03/{s}_ce").glob("*.parquet"))
    d = pd.concat([pd.read_parquet(x) for x in f], ignore_index=True).rename(columns={"ce_score": "ce3"})
    d.to_parquet(f"runs/CE03/{s}_feats.parquet", index=False)
    print(s, len(f), "files", len(d), "pairs", "mean ce3", round(float(d.ce3.mean()), 4))
PYEOF
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
PR="--cand-min-prob 0.02 --cand-ce-col ce_score --cand-ce-min 0.02"
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" --lgb-params '{"num_threads": 16}' "${@:2}" > "runs/frames2_$1.log" 2>&1 \
      || { echo "FAILED $1"; tail -15 "runs/frames2_$1.log"; }
}
run CE03_or --extra-feats runs/CE03 $PR &
if [ -f runs/OW03/train_owner.parquet ]; then run CE03OW03_or --extra-feats runs/CE03 runs/OW03 $PR & fi
wait
$PY - <<'PYEOF' | tee runs/CE03/compare.txt
import json
from pathlib import Path
def r(t):
    f = Path("runs/frames2") / t / "result.json"
    return json.loads(f.read_text()) if f.exists() else None
for base, new in (("CF_a02b02", "CE03_or"), ("CF_a02b02", "CE03OW03_or"), ("OW03_or", "CE03OW03_or")):
    b, n = r(base), r(new)
    if b and n:
        print(f"{new}: dev {n['dev_f05']:.5f} vs {base} {b['dev_f05']:.5f} -> {n['dev_f05'] - b['dev_f05']:+.5f} | "
              f"US {n['dev_by_country']['US'] - b['dev_by_country']['US']:+.5f} India "
              f"{n['dev_by_country']['India'] - b['dev_by_country']['India']:+.5f} | cand/S1 {n['cand_per_s1_dev']:.2f}")
PYEOF
aws s3 cp --only-show-errors runs/CE03/compare.txt "s3://$S3_BUCKET/$S3_PREFIX/artifacts/CE03/compare.txt"
