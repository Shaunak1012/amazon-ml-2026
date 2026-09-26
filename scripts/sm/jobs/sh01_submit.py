"""SH01 submission: average the test-like stage-2 models (runs/frames2/<tag>/stage2_model*.txt) on a test frame,
decide per S1 (expected-F0.5, one S1 per record), write + validate the two TSVs, and diff against a reference run.

    PYTHONPATH=. python scripts/sm/jobs/sh01_submit.py --frame runs/shared/frames/E020/test_frame.parquet \
        --tags SH01both SH01e12 SH01e13 SH01f30d --out submissions/sub_SH01 --ref runs/shared/E015/sub_E020
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.er_data import cache_dir, dataset_dir
from src.er_pipeline import apply_decision
from src.er_submission import write_outputs

ap = argparse.ArgumentParser()
ap.add_argument("--frame", required=True)
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rule", default="expected_f")
ap.add_argument("--t", type=float, default=0.75)
ap.add_argument("--ref", default="", help="dir with the reference run's test_probs_stage2.parquet / predict.json")
a = ap.parse_args()

Xt = pd.read_parquet(a.frame)
ft = Xt.drop(columns=["prob"])
ft["s1_prob"] = Xt.prob.to_numpy()
preds = []
for tag in a.tags:
    models = [lgb.Booster(model_file=str(p)) for p in sorted(Path("runs/frames2", tag).glob("stage2_model*.txt"))]
    cols = models[0].feature_name()
    missing = [c for c in cols if c not in ft.columns]
    assert not missing, f"{tag}: test frame lacks {missing}"
    preds.append(np.mean([m.predict(ft[cols], num_iteration=m.best_iteration) for m in models], axis=0))
    print(f"{tag}: {len(models)} models, mean prob {preds[-1].mean():.4f}", flush=True)
probs = Xt[["s1_id", "cand_id"]].assign(prob=np.mean(preds, axis=0).astype(np.float32))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
probs.to_parquet(out / "test_probs_sh01.parquet", index=False)
matches = apply_decision(probs, a.rule, a.t)
cands: dict[str, list[str]] = {}
for s, c in zip(probs.s1_id.to_numpy(), probs.cand_id.to_numpy()):
    cands.setdefault(s, []).append(c)
s1 = pd.read_parquet(cache_dir() / "test_s1_norm.parquet", columns=["entity_id", "country"])
rep = write_outputs(matches, cands, s1.entity_id.to_numpy(), out, test_dir=dataset_dir() / "test")
print("our validator:", rep)
n = s1.entity_id.map(lambda x: len(matches.get(x, ())))
info = {"tags": a.tags, "rule": a.rule, "pairs": len(probs), "nonempty_share": float((n > 0).mean()),
        "mean_matches": float(n.mean()),
        "by_country": {c: {"nonempty": float((n[s1.country == c] > 0).mean()), "mean": float(n[s1.country == c].mean())}
                       for c in s1.country.unique()}}
if a.ref and (Path(a.ref) / "test_probs_stage2.parquet").exists():
    ref = pd.read_parquet(Path(a.ref) / "test_probs_stage2.parquet")
    rj = json.loads((Path(a.ref) / "predict.json").read_text()) if (Path(a.ref) / "predict.json").exists() else {}
    rm = apply_decision(ref, rj.get("stage2_rule", "expected_f"), rj.get("stage2_t", 0.75))
    ours = {(s, c) for s, cs in matches.items() for c in cs}
    theirs = {(s, c) for s, cs in rm.items() for c in cs}
    info["vs_ref"] = {"ref_pairs": len(theirs), "our_pairs": len(ours), "only_ours": len(ours - theirs),
                      "only_ref": len(theirs - ours), "s1_changed": len({s for s, _ in ours ^ theirs})}
(out / "sh01_submit.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
