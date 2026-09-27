#!/usr/bin/env bash
# FINAL (clean): one stage-2 run -> both TSVs. Train frame E019 (clean CEs), test frame E020 with the self-trained
# ce_score_2 EXCLUDED; OW03 owner + CE03 e5-large features; candidate cascade p >= 0.05 OR ce3 >= 0.10 on train AND
# test (population features recomputed on the pruned set); test-density training (19% S1 drop).
set -euo pipefail
O=s3://$S3_BUCKET/shreyas-gpu/out/artifacts/CE03
for i in 0 1 2 3 4 5 6 7 8; do until aws s3 ls "$O/test_ce/00$i.parquet.done" >/dev/null 2>&1; do sleep 60; done; done
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
aws s3 sync --only-show-errors --exclude "*.done" "$O/test_ce/" runs/CE03dev/test_ce/
$PY -c "
import pandas as pd; from pathlib import Path
f=sorted(Path('runs/CE03dev/test_ce').glob('*.parquet'))
d=pd.concat([pd.read_parquet(x) for x in f]).rename(columns={'ce_score':'ce3'}); d.to_parquet('runs/CE03dev/test_feats.parquet', index=False)
print('test ce3:', len(f), 'files', len(d), 'pairs')"
[ -f runs/OW03/test_owner.parquet ] || { echo "missing OW03 test features"; exit 1; }
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
OUT=submissions/final_shreyas
$PY -m src.er_frames2 --frames runs/import/export/frames --test-frame runs/shared/frames/E020/test_frame.parquet \
    --train-min runs/import/export/train_min.parquet ${COMP:+--comp-cols $COMP} --tag FINAL \
    --cand-min-prob 0.05 --cand-ce-col ce3 --cand-ce-min 0.10 --extra-feats runs/CE03dev runs/OW03 \
    --drop-cols ce_score_2 --drop-s1-frac 0.19 --drop-in both --lgb-params '{"num_threads": 30}' --out $OUT
python3 data/validate_submission.py --matching $OUT/matching_results.tsv --candidate $OUT/candidate_pairs.tsv \
    --test-dir data/dataset/test --check-ids | tail -4
$PY - <<'PYEOF'
import pandas as pd
from src.er_data import cache_dir
c = pd.read_parquet(cache_dir() / "test_s1_norm.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
for f in ("matching_results", "candidate_pairs"):
    d = pd.read_csv(f"submissions/final_shreyas/{f}.tsv", sep="\t", dtype=str, keep_default_na=False)
    n = d.iloc[:, 1].map(lambda x: 0 if x == "" else len(x.split(",")))
    g = n.groupby(d.source1_entity_id.map(c)).mean().round(3).to_dict()
    print(f"{f}: mean {n.mean():.3f} per S1, empty {(n == 0).mean():.4f}, by country {g}")
PYEOF
cat runs/frames2/FINAL/predict.json
for f in matching_results.tsv candidate_pairs.tsv; do aws s3 cp --only-show-errors $OUT/$f "s3://$S3_BUCKET/$S3_PREFIX/artifacts/final_shreyas/$f"; done
aws s3 cp --only-show-errors runs/frames2/FINAL/result.json "s3://$S3_BUCKET/$S3_PREFIX/artifacts/final_shreyas/result.json"
echo FINAL READY
