#!/usr/bin/env bash
# E019 = best of E016/E017 features + French-adapted CE scores (E018) + cached stage-2 tables.
# Part 1 (after E017): dev score + train frame cache. Part 2 (after E018 France scoring): test frame + outputs.
# E018-ce/train_ce is a copy of E016-ce/train_ce (self-training changes French TEST rows only), so part 1 needs no E018.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
until [ -f runs/E017-stage2/exit.json ]; do sleep 30; done
ok E017-stage2 || { echo "CHAIN STOP: E017 failed"; exit 1; }
E17=$($PY -c "import json; print(json.load(open('runs/E015/sub_E017/stage2.json'))['dev_f05'])")
COMP=""
$PY -c "import sys; sys.exit(0 if $E17 > 0.99052 + 0.0002 else 1)" && COMP="--comp-cols cos_name name_ratio name_jw name_full_tset name_tsort"
echo "E017 dev $E17 -> comp: ${COMP:-none} ($(date +%H:%M))"
ARGS="--exp E015 --views name addr both both_ft --ce-dir runs/E015-ce runs/E018-ce --fit-folds 0 --tag E019 $COMP --frames runs/frames/E019"
$PY -m monitor.launch --run E019-dev --quiet -- $PY -m src.er_fullpass stage2 $ARGS
ok E019-dev || { echo "CHAIN STOP: E019 dev part failed"; exit 1; }
mkdir -p runs/E015/E019_dev && cp runs/E015/stage2.json runs/E015/E019_dev/
echo "E019 dev ok $(date +%H:%M): $($PY -c "import json; r=json.load(open('runs/E015/stage2.json')); print(r['dev_f05'], r['dev_by_country'])")"
until [ -f runs/E018-score/exit.json ]; do sleep 30; done
ok E018-score || { echo "CHAIN STOP: E018 France scoring failed"; exit 1; }
$PY -m monitor.launch --run E019-test --quiet -- $PY -m src.er_fullpass stage2 $ARGS --out submissions/sub_E019
ok E019-test || { echo "CHAIN STOP: E019 test part failed"; exit 1; }
mkdir -p runs/E015/sub_E019 && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E019/
echo "E019 CHAIN DONE $(date +%H:%M)"
