"""Score every SH01 arm's dev predictions under BOTH decision rules (same dev S1s, same labels)."""
import json
from pathlib import Path

import pandas as pd

from src.er_decide import decide
from src.er_frames2 import truth
from src.er_pipeline import apply_decision
from src.metrics import er_fbeta_macro

out = {}
for d in sorted(Path("runs/frames2").glob("SH01*")):
    p = pd.read_parquet(d / "dev_probs.parquet")
    y = truth(p.s1_id.unique())
    q = p[["s1_id", "cand_id", "prob"]]
    out[d.name] = {"expected_f": er_fbeta_macro(y, apply_decision(q, "expected_f", 0.75)),
                   **{f"thr_{t}": er_fbeta_macro(y, decide(q, t, True)) for t in (0.65, 0.7, 0.75, 0.8)}}
print(json.dumps(out, indent=1))
