#!/usr/bin/env bash
# OW34: is OW03 (e5-base owner model) worth adding next to OW04 (e5-large) in the E034-like regime?
# Regime = RF01 D (E023b density comp-keep 0.78, post-feature 3.83 filter p>=0.5 OR E016 CE>=0.05). Arms: none, OW03,
# OW04, OW04+OW03, two OOF partition seeds each; paired per-S1 deltas on the full 100k dev (and per dev half),
# overlap of the two owner models on dev rows (correlation, coverage), gain share of OW03 columns in the combined model.
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
D="--comp-keep 0.78 --prune-post --cand-min-prob 0.5 --cand-ce-col ce_score_2 --cand-ce-min 0.05"
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      ${COMP:+--comp-cols $COMP} --tag "$1" --lgb-params '{"num_threads": 4}' $D "${@:2}" > "runs/frames2_$1.log" 2>&1 \
      && echo "done $1 $(date +%H:%M)" || { echo "FAILED $1"; tail -15 "runs/frames2_$1.log"; }
}
for g in 0 1; do
  run OW34_none_g$g --group-seed $g &
  run OW34_ow03_g$g --group-seed $g --extra-feats runs/OW03 &
  run OW34_ow04_g$g --group-seed $g --extra-feats runs/OW04 &
  run OW34_both_g$g --group-seed $g --extra-feats runs/OW04 runs/OW03 &
done
wait
$PY - <<'PYEOF' | tee runs/frames2/ow34.txt
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.er_data import cache_dir
from src.er_refit import halves
R = Path("runs/frames2")
arms = ["none", "ow03", "ow04", "both"]
res = {(a, g): json.loads((R / f"OW34_{a}_g{g}" / "result.json").read_text()) for a in arms for g in (0, 1)}
S = {(a, g): pd.read_parquet(R / f"OW34_{a}_g{g}" / "dev_scores.parquet").f05 for a in arms for g in (0, 1)}
print("arm\tseed\tdev_f05\tUS\tIndia\toof\trule\tt")
for (a, g), r in res.items():
    print(f"{a}\t{g}\t{r['dev_f05']:.5f}\t{r['dev_by_country']['US']:.5f}\t{r['dev_by_country']['India']:.5f}\t{r['stage2_oof']:.5f}\t{r['stage2_rule']}\t{r['stage2_t']}")
A, B = halves(S[("none", 0)].index)
def pr(lbl, d):
    se = d.std(ddof=1) / np.sqrt(len(d))
    return f"{lbl}: {d.mean():+.5f} CI [{d.mean() - 1.96 * se:+.5f}, {d.mean() + 1.96 * se:+.5f}] better {(d > 0).sum()} worse {(d < 0).sum()}"
print("== paired deltas (100k dev S1; halves A/B)")
for x, y in (("none", "ow03"), ("none", "ow04"), ("none", "both"), ("ow04", "both"), ("ow03", "both")):
    for g in (0, 1):
        d = S[(y, g)] - S[(x, g)].reindex(S[(y, g)].index)
        print(pr(f"  {y} - {x} g{g}", d), "| A", f"{d[d.index.isin(A)].mean():+.5f}", "B", f"{d[d.index.isin(B)].mean():+.5f}")
    d = (S[(y, 0)] + S[(y, 1)]) / 2 - (S[(x, 0)] + S[(x, 1)]) / 2
    print(pr(f"  {y} - {x} seed-avg", d))
for a in arms:
    d = S[(a, 1)] - S[(a, 0)]
    print(pr(f"  seed noise {a} (g1 - g0)", d))
print("== overlap of the owner models on fold-0 dev rows")
folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id")
dev = set(folds.index[folds.dev])
o3 = pd.read_parquet("runs/OW03/train_owner.parquet"); o4 = pd.read_parquet("runs/OW04/train_owner.parquet")
o3, o4 = o3[o3.s1_id.isin(dev)], o4[o4.s1_id.isin(dev)]
m = o4.merge(o3, on=["s1_id", "cand_id"], how="outer", suffixes=("_4", "_3"), indicator=True)
print("  rows scored:", m["_merge"].value_counts().to_dict())
b = m[m["_merge"] == "both"]
for c in [c for c in o4.columns if c not in ("s1_id", "cand_id")]:
    x, y = b[f"{c}_4"].astype(float), b[f"{c}_3"].astype(float)
    ok = x.notna() & y.notna()
    print(f"  {c}: pearson {np.corrcoef(x[ok], y[ok])[0, 1]:.4f}, spearman {x[ok].rank().corr(y[ok].rank()):.4f}, |diff|>0.2 share {(abs(x[ok] - y[ok]) > 0.2).mean():.4f}")
if "own_p_4" in b:
    x, y = b.own_p_4 >= 0.5, b.own_p_3 >= 0.5
    print(f"  own_p>=0.5 agreement {(x == y).mean():.4f} (OW04 yes / OW03 no {(x & ~y).mean():.4f}, OW03 yes / OW04 no {(~x & y).mean():.4f})")
print("== gain share of owner columns in the combined model (both_g0)")
imp = sum(pd.Series(lgb.Booster(model_file=str(f)).feature_importance("gain"), index=lgb.Booster(model_file=str(f)).feature_name())
          for f in sorted((R / "OW34_both_g0").glob("stage2_model*.txt")))
imp = imp / imp.sum()
own = imp[[c for c in imp.index if c.startswith("own_")]].sort_values(ascending=False)
print("  " + ", ".join(f"{k} {v:.4f}" for k, v in own.items()))
print(f"  OW04 cols total {own[[c for c in own.index if not c.endswith('_OW03')]].sum():.4f}, OW03 cols total {own[[c for c in own.index if c.endswith('_OW03')]].sum():.4f}")
PYEOF
aws s3 cp --only-show-errors runs/frames2/ow34.txt "s3://$S3_BUCKET/$S3_PREFIX/artifacts/ow34.txt"
echo OW34 DONE
