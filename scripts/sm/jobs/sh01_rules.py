"""Score SH01 arms' dev predictions under both decision rules, plus averages of arms that share one dev population."""
import json
from pathlib import Path

import pandas as pd

from src.er_decide import decide
from src.er_frames2 import truth
from src.er_pipeline import apply_decision
from src.metrics import er_fbeta_macro

ENSEMBLES = {"ens_both_e12_e13": ["SH01both", "SH01e12", "SH01e13"],
             "ens_both_e12_e13_f30": ["SH01both", "SH01e12", "SH01e13", "SH01f30d"]}


def score(p: pd.DataFrame) -> dict:
    y = truth(p.s1_id.unique())
    q = p[["s1_id", "cand_id", "prob"]]
    return {"expected_f": er_fbeta_macro(y, apply_decision(q, "expected_f", 0.75)),
            **{f"thr_{t}": er_fbeta_macro(y, decide(q, t, True)) for t in (0.65, 0.7, 0.75, 0.8)}}


probs = {d.name: pd.read_parquet(d / "dev_probs.parquet") for d in sorted(Path("runs/frames2").glob("SH01*"))}
out = {k: score(v) for k, v in probs.items()}
for name, members in ENSEMBLES.items():
    if all(m in probs for m in members):
        base = probs[members[0]][["s1_id", "cand_id"]]
        same = all(len(probs[m]) == len(base) and (probs[m].cand_id.values == base.cand_id.values).all() for m in members)
        if same:
            out[name] = score(base.assign(prob=sum(probs[m].prob.to_numpy() for m in members) / len(members)))
print(json.dumps(out, indent=1))
