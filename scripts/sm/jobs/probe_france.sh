#!/usr/bin/env bash
# PROBE: sub_SH01 with every France S1 row emptied. LB(probe) vs LB(sub_SH01) gives France's true LB F0.5:
#   LB_full - LB_probe = w_F * (F_france - s_F), w_F = France share of the public subset (~0.15), s_F = France
#   singleton rate (~0.056). Same candidate file (untouched). Both validators.
set -euo pipefail
PY=.venv/bin/python
O=submissions/probe_france_sh01
mkdir -p $O
cp submissions/sub_SH01/candidate_pairs.tsv $O/
$PY - <<'PYEOF'
import pandas as pd
from src.er_data import cache_dir
m = pd.read_csv("submissions/sub_SH01/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
c = pd.read_parquet(cache_dir() / "test_s1_norm.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
fr = m.source1_entity_id.map(c).eq("France").to_numpy()
m.loc[fr, "matched_entity_ids"] = ""
m.to_csv("submissions/probe_france_sh01/matching_results.tsv", sep="\t", index=False)
print(f"emptied {fr.sum():,} France rows of {len(m):,}")
PYEOF
python3 data/validate_submission.py --matching $O/matching_results.tsv --candidate $O/candidate_pairs.tsv --test-dir data/dataset/test --check-ids | tail -3
aws s3 cp --only-show-errors $O/matching_results.tsv "s3://$S3_BUCKET/$S3_PREFIX/artifacts/probe_france_sh01/matching_results.tsv"
aws s3 cp --only-show-errors submissions/sub_SH01/matching_results.tsv "s3://$S3_BUCKET/$S3_PREFIX/artifacts/sub_SH01/matching_results.tsv"
echo PROBE READY
