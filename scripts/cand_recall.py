"""Candidate-file metrics of one stage-2 run: dev candidates/S1 and candidate recall (share of ALL dev ground-truth
pairs that survive retrieval + the candidate filter), plus test candidates/S1 from its candidate_pairs.tsv.

    python scripts/cand_recall.py --tag E023a_core --sub submissions/sub_E023a_core
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.er_data import cache_dir  # noqa: E402


def main() -> None:
    """Print one JSON line with dev cands/S1, dev candidate recall and test cands/S1."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="stage-2 --tag (reads runs/<exp>/dev_stage2_<tag>.parquet)")
    ap.add_argument("--exp-dir", default="runs/E015")
    ap.add_argument("--sub", default="", help="submission dir with candidate_pairs.tsv (test cands/S1)")
    a = ap.parse_args()
    dev = pd.read_parquet(Path(a.exp_dir) / f"dev_stage2_{a.tag}.parquet", columns=["s1_id", "cand_id", "y"])
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet")
    dev_ids = set(folds.loc[folds.dev, "s1_id"])
    gt = pd.read_parquet(cache_dir() / "train_gt_pairs.parquet", columns=["s1_id", "cand_id"])
    gt = gt[gt.s1_id.isin(dev_ids)]
    hit = gt.merge(dev[["s1_id", "cand_id"]], on=["s1_id", "cand_id"], how="inner")
    out = {"tag": a.tag, "dev_s1": len(dev_ids), "dev_cands_per_s1": round(len(dev) / len(dev_ids), 3),
           "dev_cand_recall": round(len(hit) / len(gt), 5), "dev_gt_pairs": len(gt)}
    if a.sub:
        # one row per S1, candidate ids comma-joined (empty for S1s with no candidates)
        c = pd.read_csv(Path(a.sub) / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
        n = c.iloc[:, 1].map(lambda s: len(s.split(",")) if s else 0)
        out["test_cands_per_s1"] = round(n.sum() / len(c), 3)
        out["test_s1"] = len(c)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
