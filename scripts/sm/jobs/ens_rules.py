"""Score the average of several frames2 runs' dev predictions (same dev rows required) under both decision rules."""
import json
import sys
from pathlib import Path

import pandas as pd

from src.er_decide import decide
from src.er_frames2 import truth
from src.er_pipeline import apply_decision
from src.metrics import er_fbeta_macro

tags = sys.argv[1:]
P = [pd.read_parquet(Path("runs/frames2") / t / "dev_probs.parquet") for t in tags]
base = P[0][["s1_id", "cand_id"]]
assert all(len(p) == len(base) and (p.cand_id.values == base.cand_id.values).all() for p in P), "dev rows differ"
q = base.assign(prob=sum(p.prob.to_numpy() for p in P) / len(P))
dev_ids = json.loads((Path("runs/frames2") / tags[0] / "result.json").read_text())
y = truth(q.s1_id.unique())
out = {"members": tags, "expected_f": er_fbeta_macro(y, apply_decision(q, "expected_f", 0.75)),
       **{f"thr_{t}": er_fbeta_macro(y, decide(q, t, True)) for t in (0.7, 0.75, 0.8)}}
print(json.dumps(out, indent=1))
