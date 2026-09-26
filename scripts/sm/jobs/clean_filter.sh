#!/usr/bin/env bash
# CF01: CLEAN candidate filter (no self-trained model): stage-1 p >= a OR E008 cross-encoder (ce_score) >= b.
# Team decision 26 Sep: no self-training; filter before features. Target ~4.6-4.7 cand/S1 on test.
# Then re-stage CE03's pairs with the clean superset rule (p >= 0.02 OR ce_score >= 0.02) into the SAME 3 + 9 files.
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
$PY - <<'PYEOF'
import numpy as np
import pandas as pd
from pathlib import Path
for split, f, nfiles in (("train", "runs/import/export/frames/train_frame.parquet", 3),
                         ("test", "runs/shared/frames/E020/test_frame.parquet", 9)):
    X = pd.read_parquet(f, columns=["s1_id", "cand_id", "prob", "ce_score"])
    n1 = X.s1_id.nunique()
    for a, b in ((0.02, 0.02), (0.03, 0.03), (0.05, 0.02), (0.05, 0.05), (0.1, 0.05), (0.1, 0.1)):
        k = (X.prob.to_numpy() >= a) | (np.nan_to_num(X.ce_score.to_numpy()) >= b)
        print(f"{split} p>={a} OR ce_score>={b}: {k.sum() / n1:.2f} cand/S1", flush=True)
    keep = (X.prob.to_numpy() >= 0.02) | (np.nan_to_num(X.ce_score.to_numpy()) >= 0.02)
    P = X.loc[keep, ["s1_id", "cand_id"]].reset_index(drop=True)
    d = Path("runs/CE03") / f"pairs_{split}"
    for old in d.glob("*.parquet"):
        old.unlink()
    for i, part in enumerate(np.array_split(np.arange(len(P)), nfiles)):
        P.iloc[part].to_parquet(d / f"{i:03d}.parquet", index=False)
    print(f"STAGED {split}: {len(P):,} pairs in {nfiles} files", flush=True)
PYEOF
aws s3 sync --only-show-errors --delete runs/CE03/pairs_train/ "s3://$S3_BUCKET/shreyas-gpu/data/CE03/pairs_train/"
aws s3 sync --only-show-errors --delete runs/CE03/pairs_test/ "s3://$S3_BUCKET/shreyas-gpu/data/CE03/pairs_test/"
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" --cand-min-prob "$2" --cand-ce-col ce_score --cand-ce-min "$3" \
      --lgb-params '{"num_threads": 8}' > "runs/frames2_$1.log" 2>&1 || { echo "FAILED $1"; tail -8 "runs/frames2_$1.log"; }
}
export -f run; export PY COMP
for v in "CF_a02b02 0.02 0.02" "CF_a03b03 0.03 0.03" "CF_a05b02 0.05 0.02" "CF_a05b05 0.05 0.05"; do echo "$v"; done \
  | xargs -P 4 -L 1 bash -c 'run $0 $1 $2'
$PY - <<'PYEOF' | tee runs/frames2/clean_filter.tsv
import json
from pathlib import Path
print("tag\tmin_prob\tce_min\tcand_per_s1\trecall\tdev_f05\tUS\tIndia")
for t in ("CS_k15", "CS_or02c02", "CF_a02b02", "CF_a03b03", "CF_a05b02", "CF_a05b05"):
    f = Path("runs/frames2") / t / "result.json"
    if f.exists():
        r = json.loads(f.read_text())
        print(f"{t}\t{r['cand_min_prob']}\t{r.get('cand_ce_min', '')}\t{r['cand_per_s1_dev']:.2f}\t{r['cand_recall_dev']:.5f}"
              f"\t{r['dev_f05']:.5f}\t{r['dev_by_country']['US']:.5f}\t{r['dev_by_country']['India']:.5f}")
PYEOF
aws s3 cp --only-show-errors runs/frames2/clean_filter.tsv "s3://$S3_BUCKET/$S3_PREFIX/artifacts/clean_filter.tsv"
