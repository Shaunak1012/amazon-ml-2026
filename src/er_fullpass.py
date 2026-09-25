"""Full-population stage 1 + stage 2 (fixes the cross-S1 feature mismatch, docs/DECISIONS.md 2026-09-25).

Cross-S1 features (a record's competing S1s) depend on how many S1s are scored, so every split must be scored in
full: all 2.2M train S1 (out-of-fold) and all 1.7M test S1. Then stage 2 is trained/validated/applied in the same
regime as test.

    python -m src.er_fullpass stage1 --exp E007 --train-s1 200000          # ~2 h: fit, score all train + test
    python -m src.er_fullpass stage2 --exp E007 --out submissions/sub03    # validate on dev, then predict test

Stage-1 fitting uses S1s from folds 1-4 only, so dev (fold 0) stays untouched: a train S1 in fold g is scored by
the model that excluded fold g; fold-0 and test S1s are scored by the mean of all fold models.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.er_data import cache_dir
from src.er_model import predict, train_oof
from src.er_pipeline import Split, apply_decision, choose_decision, featurize, log, retrieve, topk_mask
from src.er_stage2 import cluster_features, competition_features
from src.metrics import er_fbeta_macro

COMP = ("s2_other_s1_best", "s2_margin_vs_other_s1")


def run_dir(exp: str) -> Path:
    d = Path("runs") / exp
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_stage1(a: argparse.Namespace) -> None:
    from monitor import Heartbeat

    t0 = time.time()
    rd = run_dir(a.exp)
    hb = Heartbeat(os.environ.get("RUN_ID", a.exp), total_steps=25, every_steps=1)
    d = Split("train", a.model)
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id").reindex(d.s1.entity_id)
    fold_arr = folds.fold.to_numpy()
    rng = np.random.default_rng(42)
    fit_mask = np.zeros(len(d.s1), bool)
    fit_mask[rng.choice(np.flatnonzero(fold_arr != 0), size=a.train_s1, replace=False)] = True
    X = featurize(d, retrieve(d, fit_mask, a.k))
    truth = set(zip(d.gt.s1_id, d.gt.cand_id))
    X["y"] = [(s, c) in truth for s, c in zip(X.s1_id, X.cand_id)]
    groups = folds.fold.reindex(X.s1_id).to_numpy()
    _, models = train_oof(X, X.y.to_numpy().astype(int), groups, {"learning_rate": a.lr},
                          num_boost_round=a.rounds, early_stopping=50)
    fold_ids = list(np.unique(groups))              # models[i] excluded fold fold_ids[i]
    for i, m in enumerate(models):
        m.save_model(str(rd / f"stage1_fold{fold_ids[i]}.txt"))
    log(f"stage 1 fitted on {fit_mask.sum():,} S1 ({len(X):,} pairs), folds {fold_ids} [{time.time() - t0:.0f}s]")
    del X
    hb.step(1, force=True, stage="score train")

    def score(split_obj: Split, name: str, fold_of_s1: np.ndarray | None, labels: set | None, step0: int) -> None:
        out = rd / f"{name}_chunks"
        out.mkdir(exist_ok=True)
        n = min(len(split_obj.s1), a.max_s1) if a.max_s1 else len(split_obj.s1)   # --max-s1: smoke tests only
        for ci, start in enumerate(range(0, n, a.chunk)):
            path = out / f"{ci:03d}.parquet"
            if path.exists():
                continue                                      # resumable
            mask = np.zeros(len(split_obj.s1), bool)
            mask[start:min(start + a.chunk, n)] = True
            Xc = featurize(split_obj, retrieve(split_obj, mask, a.k))
            prob = predict(models, Xc)                        # default: mean of fold models
            if fold_of_s1 is not None:                        # train: out-of-fold model per S1's fold
                f = pd.Series(fold_of_s1, index=split_obj.s1.entity_id).reindex(Xc.s1_id).to_numpy()
                cols = models[0].feature_name()
                for i, fid in enumerate(fold_ids):
                    sel = f == fid
                    if sel.any():
                        prob[sel] = models[i].predict(Xc.loc[sel, cols], num_iteration=models[i].best_iteration)
            Xc["prob"] = prob.astype(np.float32)
            Xc = Xc[topk_mask(Xc, a.prefilter)].reset_index(drop=True)
            if labels is not None:
                Xc["y"] = [(s, c) in labels for s, c in zip(Xc.s1_id, Xc.cand_id)]
            Xc.to_parquet(path, index=False)
            log(f"{name} chunk {ci + 1}: kept {len(Xc):,} pairs [{time.time() - t0:.0f}s]")
            hb.step(step0 + ci, force=True, stage=f"score {name}")

    score(d, "train", fold_arr, truth, 2)
    del d
    te = Split("test", a.model)
    score(te, "test", None, None, 14)
    (rd / "stage1.json").write_text(json.dumps({"fold_ids": [int(x) for x in fold_ids], "k": a.k,
                                                "prefilter": a.prefilter, "train_s1": a.train_s1,
                                                "runtime_s": round(time.time() - t0)}), encoding="utf-8")
    hb.finish("completed")
    log(f"done [{time.time() - t0:.0f}s]")


def load_chunks(rd: Path, name: str, columns=None) -> list[pd.DataFrame]:
    return [pd.read_parquet(p, columns=columns) for p in sorted((rd / f"{name}_chunks").glob("*.parquet"))]


def with_ce(chunks: list[pd.DataFrame], ce_dir: str, split: str) -> list[pd.DataFrame]:
    """Attach cross-encoder scores (src.er_crossenc score output: one parquet per chunk, same row order)."""
    if not ce_dir:
        return chunks
    files = sorted((Path(ce_dir) / f"{split}_ce").glob("*.parquet"))
    if len(files) != len(chunks):
        raise ValueError(f"{len(files)} CE files for {len(chunks)} {split} chunks in {ce_dir}")
    out = []
    for c, f in zip(chunks, files):
        ce = pd.read_parquet(f)
        if len(ce) != len(c) or not (ce.s1_id.to_numpy() == c.s1_id.to_numpy()).all() \
                or not (ce.cand_id.to_numpy() == c.cand_id.to_numpy()).all():
            raise ValueError(f"CE rows misaligned with {f.name}")
        out.append(c.assign(ce_score=ce.ce_score.to_numpy(np.float32)))
    return out


def stage2_frames(split_obj: Split, chunks: list[pd.DataFrame], keep_s1: set | None) -> pd.DataFrame:
    """Stage-2 design matrix: sibling feats per chunk (chunks partition S1) + competition feats over ALL chunks."""
    allp = pd.concat([c[["s1_id", "cand_id", "prob"]] for c in chunks], ignore_index=True)
    comp = competition_features(allp)
    parts, off = [], 0
    emb = {v: split_obj.embp[v] for v in split_obj.views}
    for c in chunks:
        n = len(c)
        sel = np.ones(n, bool) if keep_s1 is None else c.s1_id.isin(keep_s1).to_numpy()
        if sel.any():
            cs = c[sel].reset_index(drop=True)
            F = cluster_features(cs[["s1_id", "cand_id", "prob"]], split_obj.right, emb)
            for k in COMP:
                F[k] = comp[k][off:off + n][sel]
            parts.append(pd.concat([cs, F], axis=1))
        off += n
    return pd.concat(parts, ignore_index=True)


def cmd_stage2(a: argparse.Namespace) -> None:
    from src.er_data import dataset_dir
    from src.er_submission import write_outputs

    t0 = time.time()
    rd = run_dir(a.exp)
    d = Split("train", a.model)
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id")
    rng = np.random.default_rng(7)
    pool = folds.index[folds.fold.isin(a.fit_folds)].to_numpy()   # e.g. 3 4 when the CE trained on folds 1-2
    fit_ids = set(rng.choice(pool, size=min(a.train_s1, len(pool)), replace=False))
    dev_ids = set(folds.index[folds.dev])
    X = stage2_frames(d, with_ce(load_chunks(rd, "train"), a.ce_dir, "train"), fit_ids | dev_ids)
    log(f"stage-2 frame {X.shape} [{time.time() - t0:.0f}s]")
    is_dev = X.s1_id.isin(dev_ids).to_numpy()
    Xtr, Xdev = X[~is_dev].reset_index(drop=True), X[is_dev].reset_index(drop=True)
    y_tr, y_dev = d.truth(sorted(fit_ids)), d.truth(sorted(dev_ids))
    # stage-1-only reference on the same (full-population, top-k) candidates
    p1_tr = Xtr[["s1_id", "cand_id", "prob"]]
    rule1, t1, _ = choose_decision(p1_tr, y_tr)
    res = {"exp": a.exp, "stage1_dev_f05": er_fbeta_macro(y_dev, apply_decision(Xdev[["s1_id", "cand_id", "prob"]], rule1, t1))}
    feat = Xtr.drop(columns=["prob"])
    feat["s1_prob"] = Xtr.prob.to_numpy()
    groups = folds.fold.reindex(Xtr.s1_id).to_numpy()
    oof, models = train_oof(feat, Xtr.y.to_numpy().astype(int), groups, {"learning_rate": a.lr},
                            num_boost_round=a.rounds, early_stopping=50)
    fdev = Xdev.drop(columns=["prob"])
    fdev["s1_prob"] = Xdev.prob.to_numpy()
    p_tr = Xtr[["s1_id", "cand_id"]].assign(prob=oof)
    p_dev = Xdev[["s1_id", "cand_id"]].assign(prob=predict(models, fdev))
    rule, t, f = choose_decision(p_tr, y_tr)
    res.update({"stage2_rule": rule, "stage2_t": t, "stage2_oof": f,
                "dev_f05": er_fbeta_macro(y_dev, apply_decision(p_dev, rule, t))})
    ctry = d.s1.set_index("entity_id").country
    pred = apply_decision(p_dev, rule, t)
    res["dev_by_country"] = {c: er_fbeta_macro({k: v for k, v in y_dev.items() if ctry[k] == c}, pred)
                             for c in ("US", "India")}
    log(json.dumps(res))
    (rd / "stage2.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    del d, X, Xtr, Xdev
    if not a.out:
        return
    te = Split("test", a.model)
    Xt = stage2_frames(te, with_ce(load_chunks(rd, "test"), a.ce_dir, "test"), None)
    ft = Xt.drop(columns=["prob"])
    ft["s1_prob"] = Xt.prob.to_numpy()
    probs = Xt[["s1_id", "cand_id"]].assign(prob=predict(models, ft).astype(np.float32))
    probs.to_parquet(rd / "test_probs_stage2.parquet", index=False)
    matches = apply_decision(probs, rule, t)
    cands: dict[str, list[str]] = {}
    for s, c in zip(probs.s1_id.to_numpy(), probs.cand_id.to_numpy()):
        cands.setdefault(s, []).append(c)
    write_outputs(matches, cands, te.s1.entity_id.to_numpy(), Path(a.out), test_dir=dataset_dir() / "test")
    s1 = te.s1.entity_id.to_numpy()
    info = {**res, "test_pairs": len(probs), "nonempty_share": sum(1 for v in matches.values() if v) / len(s1),
            "mean_matches": float(np.mean([len(matches.get(x, ())) for x in s1])), "runtime_s": round(time.time() - t0)}
    (rd / "predict.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(json.dumps(info))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m src.er_fullpass")
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage1", "stage2"):
        p = sp.add_parser(name)
        p.add_argument("--exp", required=True)
        p.add_argument("--model", default="small")
        p.add_argument("--lr", type=float, default=0.1)
        p.add_argument("--rounds", type=int, default=1000)
    s1 = sp.choices["stage1"]
    s1.add_argument("--k", type=int, default=10)
    s1.add_argument("--train-s1", type=int, default=200_000)
    s1.add_argument("--prefilter", type=int, default=15)
    s1.add_argument("--chunk", type=int, default=200_000)
    s1.add_argument("--max-s1", type=int, default=0, help="score only the first N S1 per split (smoke test)")
    s2 = sp.choices["stage2"]
    s2.add_argument("--train-s1", type=int, default=300_000)
    s2.add_argument("--fit-folds", type=int, nargs="+", default=[1, 2, 3, 4], help="folds whose S1s fit stage 2 (use 3 4 if the CE trained on folds 1-2)")
    s2.add_argument("--ce-dir", default="", help="run dir with train_ce/ and test_ce/ score parquets")
    s2.add_argument("--out", default="")
    a = ap.parse_args()
    {"stage1": cmd_stage1, "stage2": cmd_stage2}[a.cmd](a)


if __name__ == "__main__":
    main()
