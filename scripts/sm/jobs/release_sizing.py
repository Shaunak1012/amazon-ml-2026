"""JD01 sizing: on real test stage-2 probabilities, how many records does the greedy rule lose?

Greedy rule (er_decide): each record goes to its highest-prob S1, then each S1 picks its subset. A record its top S1
rejects is lost for every S1. Count those whose SECOND-best S1 is confident (p2 >= 0.5 / 0.7 / 0.9), by country,
and how many S1s would gain a match. Upper bound on what a release loop can recover (no labels on test).

    PYTHONPATH=. python scripts/sm/jobs/release_sizing.py runs/shared/E015/sub_E016
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.er_data import cache_dir
from src.er_pipeline import apply_decision

d = Path(sys.argv[1])
P = pd.read_parquet(d / "test_probs_stage2.parquet", columns=["s1_id", "cand_id", "prob"])
mf = next((d / f for f in ("predict.json", "stage2.json") if (d / f).exists()), None)
meta = json.loads(mf.read_text()) if mf else {}
rule, t = meta.get("stage2_rule", "expected_f"), meta.get("stage2_t", 0.75)
print(f"{len(P):,} pairs, rule {rule} t {t}", flush=True)
matches = apply_decision(P, rule, t)
sel = {(s, c) for s, cs in matches.items() for c in cs}
Ps = P.sort_values(["cand_id", "prob"], ascending=[True, False], kind="stable")
first = Ps.drop_duplicates("cand_id", keep="first").set_index("cand_id")
rest = Ps.groupby("cand_id", sort=False).nth(1)                  # each record's second-best S1
second = rest.set_index("cand_id")[["s1_id", "prob"]].rename(columns={"s1_id": "s1_2", "prob": "p2"})
R = first.join(second, how="left")
R["selected"] = [(s, c) in sel for s, c in zip(R.s1_id.to_numpy(), R.index.to_numpy())]
ctry = pd.read_parquet(cache_dir() / "test_s1_norm.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
R["country"] = ctry.reindex(R.s1_id).to_numpy()
lost = R[~R.selected & R.p2.notna()]
out = {"records_with_candidates": len(R), "records_selected": int(R.selected.sum()),
       "records_rejected_by_top_with_2nd_s1": len(lost)}
for th in (0.5, 0.7, 0.9):
    x = lost[lost.p2 >= th]
    out[f"lost_with_p2>={th}"] = {"records": len(x), "s1_affected": int(x.s1_2.nunique()),
                                  "by_country": x.country.value_counts().to_dict()}
out["s1_total"] = int(ctry.shape[0])
print(json.dumps(out, indent=2))
