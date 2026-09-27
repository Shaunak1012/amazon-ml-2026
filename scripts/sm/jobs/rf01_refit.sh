#!/usr/bin/env bash
# RF01: does stage 2 gain from training on the 100k dev S1s too (final model sees 400k instead of 300k fold-0 S1s)?
# Candidate rows unchanged. src/er_refit.py (via er_frames2 --refit-check):
#   D = E034-like regime (Shaunak's E023b density comp-keep 0.78, post-feature 3.83 filter p>=0.5 OR E016 CE>=0.05,
#       + OW04), dev halves A and B (train with the other half added, score this half, paired)
#   T = our final-v2 regime (CE03 cascade filter, + CE03 + OW04), dev half A (replication in a second pipeline) and
#       final mode (dev folded into fit) with test decision drift per country (France at the 0.85 unseen threshold)
set -euo pipefail
bash scripts/sm/pull_code.sh
PY=.venv/bin/python
COMP=$($PY -c "import json; print(' '.join(json.load(open('runs/import/export/frames/frames.json')).get('comp_cols', [])))")
D="--comp-keep 0.78 --prune-post --cand-min-prob 0.5 --cand-ce-col ce_score_2 --cand-ce-min 0.05 --extra-feats runs/OW04"
T="--cand-min-prob 0.05 --cand-ce-col ce3 --cand-ce-min 0.10 --extra-feats runs/CE03dev runs/OW04 --drop-cols ce_score_2"
run() {
  $PY -m src.er_frames2 --frames runs/import/export/frames --train-min runs/import/export/train_min.parquet \
      --test-frame runs/shared/frames/E020/test_frame.parquet ${COMP:+--comp-cols $COMP} --tag "$1" \
      --lgb-params '{"num_threads": 8}' "${@:2}" > "runs/frames2_$1.log" 2>&1 \
      && echo "done $1 $(date +%H:%M)" || { echo "FAILED $1"; tail -15 "runs/frames2_$1.log"; }
}
run RF01_D_A $D --refit-check A &
run RF01_D_B $D --refit-check B &
run RF01_T_A $T --refit-check A &
run RF01_T_final $T --refit-check final &
wait
$PY - <<'PYEOF' | tee runs/frames2/rf01_refit.txt
import json
from pathlib import Path
import numpy as np
import pandas as pd
R = Path("runs/frames2")
def load(t):
    f = R / t / "refit.json"
    return json.loads(f.read_text()) if f.exists() else None
for t in ("RF01_D_A", "RF01_D_B", "RF01_T_A"):
    r = load(t)
    if r is None:
        print(t, "MISSING"); continue
    print(f"== {t}: eval {r['n_eval']:,} dev S1, fit base {r['n_fit_base']:,} / add {r['n_fit_add']:,}, cand/S1 {r['cand_per_s1_dev']:.3f}")
    print("arm\tdev_f05\tUS\tIndia\toof\toof-dev\trule\tt\tbest_dev_t\tf@best_t\tlogloss\tbrier\tmean_p\tpos_rate\tmatch/S1\tempty\titers")
    for k, v in r["arms"].items():
        it = v["best_iter"] if v["best_iter"] is not None else v["rounds"]
        print(f"{k}\t{v['dev_f05']:.5f}\t{v['US']:.5f}\t{v['India']:.5f}\t{v['oof_f05']:.5f}\t{v['oof_minus_dev']:+.5f}\t{v['rule']}\t{v['t']}"
              f"\t{v['best_dev_t']}\t{v['f_at_best_dev_t']:.5f}\t{v['logloss']:.5f}\t{v['brier']:.5f}\t{v['mean_prob']:.4f}\t{v['pos_rate']:.4f}"
              f"\t{v['matches_per_s1']:.4f}\t{v['empty_share']:.4f}\t{it}")
    for k, v in r["paired"].items():
        print(f"  {k}: {v['delta']:+.5f} CI [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] better {v['n_better']} worse {v['n_worse']}")
# D: pool halves A + B -> every dev S1 scored once, paired over 100k S1
sc = [R / t / "dev_scores.parquet" for t in ("RF01_D_A", "RF01_D_B")]
if all(f.exists() for f in sc):
    S = pd.concat([pd.read_parquet(f) for f in sc])
    print(f"== D pooled over halves A+B ({len(S):,} dev S1)")
    for arm in S.columns:
        print(f"  {arm}: {S[arm].mean():.5f}")
    for a, b in (("base_g0", "base_g1"), ("base_g0", "add_g0"), ("base_g1", "add_g1"), ("base_g0", "full_base_x1.00"),
                 ("base_g0", "full_add_x1.00"), ("base_g0", "full_add_x1.25"), ("add_g0", "full_add_x1.00")):
        d = (S[b] - S[a]).to_numpy(); se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {b} - {a}: {d.mean():+.5f} CI [{d.mean() - 1.96 * se:+.5f}, {d.mean() + 1.96 * se:+.5f}] better {(d > 0).sum()} worse {(d < 0).sum()}")
r = load("RF01_T_final")
if r:
    print("== T final mode: test decision drift vs base_g0 (France at the 0.85 unseen threshold)")
    for k, v in r["arms"].items():
        print(f"  {k}: rule {v['rule']} t {v['t']} oof {v['oof_f05']:.5f} iters {v.get('best_iter') or v.get('rounds')}")
    for k, row in r["drift_vs_base_g0"].items():
        for c, x in row.items():
            if c == "pairs_mean_prob":
                print(f"  {k}: mean test prob base {x[0]:.4f} arm {x[1]:.4f}"); continue
            print(f"  {k} {c}: S1 changed {x['s1_changed']:.4%}, matches/S1 {x['matches_base']:.4f} -> {x['matches_arm']:.4f} "
                  f"({x['matches_rel_change']:+.3%}), empty {x['empty_base']:.4f} -> {x['empty_arm']:.4f}")
PYEOF
for t in RF01_D_A RF01_D_B RF01_T_A RF01_T_final; do
  [ -f runs/frames2/$t/refit.json ] && aws s3 cp --only-show-errors runs/frames2/$t/refit.json "s3://$S3_BUCKET/$S3_PREFIX/artifacts/rf01/$t.json"
done
aws s3 cp --only-show-errors runs/frames2/rf01_refit.txt "s3://$S3_BUCKET/$S3_PREFIX/artifacts/rf01_refit.txt"
echo RF01 DONE
