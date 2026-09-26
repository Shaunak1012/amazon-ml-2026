"""Stage 2 from cached design matrices (CPU box: no GPU, no embeddings needed).

`src.er_fullpass stage2 --frames DIR` caches train_frame.parquet (fit + dev S1 rows, all stage-1 + CE + cluster +
competition + rarity features, y) and test_frame.parquet. This module reruns stage 2 on them with the SAME training,
decision and metric code (train_oof, choose_decision, er_fbeta_macro), so its dev scores compare 1:1 with
er_fullpass runs. It also re-derives the population-dependent columns for the distractor simulation:

    test has ~2.3 unmatched pool records per S1, train ~1.2 (docs/SHREYAS_LOG.md). Dropping ~19% of train S1s makes
    their matched records unmatched distractors; the record-level competition features (best OTHER S1) and the name
    rarity counts are then recomputed over the reduced population. Per-S1 cluster features do not depend on other
    S1s and are reused as cached.

    python -m src.er_frames2 --frames runs/frames/E019 --train-min runs/import/train_min.parquet \
        --drop-s1-frac 0.19 --drop-in both --tag SH01both [--out submissions/sub_SH01]

Limitation: stage-1 features with record-side rank context (ctx_rank_in_cand) were computed with every S1 present and
are not recomputed (they are chunk-local in stage 1 anyway).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.er_data import cache_dir, dataset_dir
from src.er_fullpass import lgb_params, name_rarity
from src.er_model import predict, train_oof
from src.er_pipeline import apply_decision, choose_decision, log
from src.er_stage2 import competition_features
from src.metrics import er_fbeta_macro

RAR = ("rar_s1_same_name_s1", "rar_s1_same_name_pool", "rar_s1_min_tok_df",
       "rar_c_same_name_pool", "rar_c_same_name_s1", "rar_c_min_tok_df")


def names(split: str) -> SimpleNamespace:
    """left/right frames (country, name_core) indexed by entity_id, as name_rarity expects."""
    cols = ["entity_id", "country", "name_core"]
    n = [pd.read_parquet(cache_dir() / f"{split}_s{k}_norm.parquet", columns=cols) for k in (1, 2, 3)]
    return SimpleNamespace(left=n[0].set_index("entity_id"),
                           right=pd.concat(n[1:], ignore_index=True).set_index("entity_id"))


def truth(s1_ids) -> dict[str, set[str]]:
    gt = pd.read_parquet(cache_dir() / "train_gt_pairs.parquet")
    gt = gt[gt.s1_id.isin(set(s1_ids))]
    y = {s: set() for s in s1_ids}
    for s, c in zip(gt.s1_id.to_numpy(), gt.cand_id.to_numpy()):
        y[s].add(c)
    return y


def repopulate(X: pd.DataFrame, allp: pd.DataFrame, nm: SimpleNamespace, drop: set,
               comp_cols: tuple[str, ...]) -> pd.DataFrame:
    """Recompute competition + rarity columns of X's rows in the population without the `drop` S1s."""
    if not drop:
        return X
    P = allp[~allp.s1_id.isin(drop)].reset_index(drop=True)
    comp = competition_features(P)
    for col in comp_cols:
        comp.update(competition_features(P, col))
    C = P[["s1_id", "cand_id"]].assign(**comp)
    X = X.drop(columns=[c for c in comp if c in X.columns])
    X = X.merge(C, on=["s1_id", "cand_id"], how="left", validate="one_to_one")
    rl, rr = name_rarity(nm, drop)
    for c in rl.columns:
        X[c] = rl[c].reindex(X.s1_id).to_numpy()
    for c in rr.columns:
        X[c] = rr[c].reindex(X.cand_id).to_numpy()
    return X


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m src.er_frames2")
    ap.add_argument("--frames", required=True, help="dir with train_frame.parquet (+ test_frame.parquet for --out)")
    ap.add_argument("--train-min", default="", help="parquet [s1_id, cand_id, prob, comp cols] of ALL train chunks")
    ap.add_argument("--comp-cols", nargs="*", default=[])
    ap.add_argument("--drop-s1-frac", type=float, default=0.0)
    ap.add_argument("--drop-in", choices=["both", "eval"], default="both")
    ap.add_argument("--drop-seed", type=int, default=11)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--lgb-params", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    rd = Path("runs") / "frames2" / a.tag
    rd.mkdir(parents=True, exist_ok=True)

    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id")
    X = pd.read_parquet(Path(a.frames) / "train_frame.parquet")
    drop: set = set()
    if a.drop_s1_frac > 0:
        ids = folds.index.to_numpy()
        drop = set(np.random.default_rng(a.drop_seed).choice(ids, int(round(a.drop_s1_frac * len(ids))), replace=False))
    drop_fit = drop if a.drop_in == "both" else set()
    is_dev = folds.dev.reindex(X.s1_id).to_numpy(bool)
    Xtr = X[~is_dev & ~X.s1_id.isin(drop_fit).to_numpy()].reset_index(drop=True)
    Xdev = X[is_dev & ~X.s1_id.isin(drop).to_numpy()].reset_index(drop=True)
    del X
    if drop:
        allp = pd.read_parquet(a.train_min, columns=["s1_id", "cand_id", "prob", *a.comp_cols])
        nm = names("train")
        Xtr = repopulate(Xtr, allp, nm, drop_fit, tuple(a.comp_cols))
        Xdev = repopulate(Xdev, allp, nm, drop, tuple(a.comp_cols))
        del allp
    log(f"frames: fit {Xtr.shape}, dev {Xdev.shape}, dropped {len(drop):,} S1 ({a.drop_in}) [{time.time() - t0:.0f}s]")

    fit_ids, dev_ids = Xtr.s1_id.unique(), Xdev.s1_id.unique()
    y_all = truth(np.concatenate([fit_ids, dev_ids]))
    y_tr, y_dev = {k: y_all[k] for k in fit_ids}, {k: y_all[k] for k in dev_ids}
    rule1, t1, _ = choose_decision(Xtr[["s1_id", "cand_id", "prob"]], y_tr)
    res = {"tag": a.tag, "drop_s1_frac": a.drop_s1_frac, "drop_in": a.drop_in, "n_fit": len(fit_ids),
           "n_dev": len(dev_ids),
           "stage1_dev_f05": er_fbeta_macro(y_dev, apply_decision(Xdev[["s1_id", "cand_id", "prob"]], rule1, t1))}
    feat = Xtr.drop(columns=["prob"])
    feat["s1_prob"] = Xtr.prob.to_numpy()
    groups = folds.fold.reindex(Xtr.s1_id).to_numpy()
    if len(np.unique(groups)) < 2:                     # same internal grouping as er_fullpass stage 2
        u = pd.Index(Xtr.s1_id.unique())
        groups = pd.Series(np.random.default_rng(0).integers(0, 4, len(u)), index=u).reindex(Xtr.s1_id).to_numpy()
    params = lgb_params(a)
    oof, models = train_oof(feat, Xtr.y.to_numpy().astype(int), groups, params, num_boost_round=a.rounds,
                            early_stopping=50)
    fdev = Xdev.drop(columns=["prob"])
    fdev["s1_prob"] = Xdev.prob.to_numpy()
    p_tr = Xtr[["s1_id", "cand_id"]].assign(prob=oof)
    p_dev = Xdev[["s1_id", "cand_id"]].assign(prob=predict(models, fdev))
    p_dev.assign(y=Xdev.y.to_numpy()).to_parquet(rd / "dev_probs.parquet", index=False)
    rule, t, f = choose_decision(p_tr, y_tr)
    pred = apply_decision(p_dev, rule, t)
    ctry = names("train").left.country
    res.update({"stage2_rule": rule, "stage2_t": t, "stage2_oof": f, "dev_f05": er_fbeta_macro(y_dev, pred),
                "dev_by_country": {c: er_fbeta_macro({k: v for k, v in y_dev.items() if ctry[k] == c}, pred)
                                   for c in ("US", "India")}})
    for i, m in enumerate(models):
        m.save_model(str(rd / f"stage2_model{i}.txt"))
    log(json.dumps(res))
    (rd / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    del Xtr, Xdev, feat, fdev
    if not a.out:
        return
    from src.er_submission import write_outputs
    Xt = pd.read_parquet(Path(a.frames) / "test_frame.parquet")
    ft = Xt.drop(columns=["prob"])
    ft["s1_prob"] = Xt.prob.to_numpy()
    probs = Xt[["s1_id", "cand_id"]].assign(prob=predict(models, ft).astype(np.float32))
    probs.to_parquet(rd / "test_probs.parquet", index=False)
    matches = apply_decision(probs, rule, t)
    cands: dict[str, list[str]] = {}
    for s, c in zip(probs.s1_id.to_numpy(), probs.cand_id.to_numpy()):
        cands.setdefault(s, []).append(c)
    s1 = pd.read_parquet(cache_dir() / "test_s1_norm.parquet", columns=["entity_id"]).entity_id.to_numpy()
    write_outputs(matches, cands, s1, Path(a.out), test_dir=dataset_dir() / "test")
    info = {**res, "test_pairs": len(probs), "nonempty_share": sum(1 for v in matches.values() if v) / len(s1),
            "mean_matches": float(np.mean([len(matches.get(x, ())) for x in s1])), "runtime_s": round(time.time() - t0)}
    (rd / "predict.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(json.dumps(info))


if __name__ == "__main__":
    main()
