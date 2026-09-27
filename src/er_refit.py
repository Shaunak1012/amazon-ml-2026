"""Does stage 2 gain from training on more S1s, and does it hurt anything? (RF01, run via er_frames2 --refit-check)

er_fullpass stage 2 predicts test with the 4 OOF models of the fit S1s (each trains on ~3/4 of the 300k fold-0 fit
S1s); the 100k dev S1s only score. Two ways to use them in the final model, both on the same candidate rows
(candidate_pairs.tsv unchanged):
  add   put the dev S1s into the fit pool; same OOF procedure (4 models, early stopping, rule + threshold on OOF)
  full  refit one model on every S1 with a fixed round count (mean OOF best iteration x scale); rule from the OOF run

Measured without scoring on training rows: dev is split into fixed halves (seed 5); an arm trains with the OTHER half
added and is scored on this half, paired with the base arm on the same rows. Two partition seeds per arm give the
noise floor. Detriment checks: OOF-vs-dev gap, best iterations, calibration on dev rows (log loss, Brier, mean prob vs
positive rate), threshold transfer (F at the OOF-chosen threshold vs the dev-optimal one), matches per S1, empty share,
US/India; in final mode, test decision drift per country (France unseen) against the base arm's seed-to-seed drift.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.er_decide import decide
from src.er_model import DEFAULT_PARAMS, feature_cols, predict, train_oof
from src.er_pipeline import apply_decision, choose_decision, log
from src.metrics import entity_fbeta

GRID = np.round(np.arange(0.30, 0.96, 0.05), 2)


def halves(ids, seed: int = 5) -> tuple[set, set]:
    """Fixed random halves of the dev S1s."""
    ids = np.sort(np.asarray(list(ids)))
    perm = np.random.default_rng(seed).permutation(len(ids))
    h = len(ids) // 2
    return set(ids[perm[:h]]), set(ids[perm[h:]])


def design(X: pd.DataFrame) -> pd.DataFrame:
    f = X.drop(columns=["prob"])
    f["s1_prob"] = X.prob.to_numpy()
    return f


def fit_oof(X: pd.DataFrame, y_fit: dict, params: dict, rounds: int, seed: int) -> dict:
    """er_frames2 stage 2 exactly (4 random S1 groups, early stopping 50, rule on OOF) with partition seed `seed`."""
    u = pd.Index(X.s1_id.unique())
    groups = pd.Series(np.random.default_rng(seed).integers(0, 4, len(u)), index=u).reindex(X.s1_id).to_numpy()
    oof, models = train_oof(design(X), X.y.to_numpy().astype(int), groups, params, num_boost_round=rounds,
                            early_stopping=50)
    rule, t, f = choose_decision(X[["s1_id", "cand_id"]].assign(prob=oof), y_fit)
    return {"models": models, "rule": rule, "t": t, "oof_f05": f, "best_iter": [m.best_iteration for m in models]}


def fit_full(X: pd.DataFrame, params: dict, n_rounds: int) -> lgb.Booster:
    """One model on every row, fixed rounds (no holdout left for early stopping)."""
    f = design(X)
    cols = feature_cols(f)
    return lgb.train({**DEFAULT_PARAMS, **params}, lgb.Dataset(f[cols], label=X.y.to_numpy().astype(int)), n_rounds)


def per_s1(pred: dict, y: dict) -> pd.Series:
    return pd.Series({k: entity_fbeta(set(v), set(pred.get(k, ())), 0.5) for k, v in y.items()})


def evaluate(p: pd.DataFrame, y: dict, rule: str, t: float, ctry: pd.Series) -> tuple[pd.Series, dict]:
    """p: dev rows [s1_id, cand_id, y, prob]. Per-S1 scores + summary with the detriment diagnostics."""
    q = p[["s1_id", "cand_id", "prob"]]
    pred = apply_decision(q, rule, t)
    s = per_s1(pred, y)
    c = ctry.reindex(s.index).to_numpy()
    yv, pr = p.y.to_numpy().astype(float), np.clip(p.prob.to_numpy(), 1e-7, 1 - 1e-7)
    thr = {float(g): float(per_s1(decide(q, g, True), y).mean()) for g in GRID}
    best_t = max(thr, key=thr.get)
    n = np.array([len(pred.get(k, ())) for k in y])
    return s, {"dev_f05": float(s.mean()), "US": float(s[c == "US"].mean()), "India": float(s[c == "India"].mean()),
               "rule": rule, "t": t, "f_at_t_grid": thr.get(round(t, 2)), "best_dev_t": best_t,
               "f_at_best_dev_t": thr[best_t],
               "logloss": float(-np.mean(yv * np.log(pr) + (1 - yv) * np.log(1 - pr))),
               "brier": float(np.mean((pr - yv) ** 2)), "mean_prob": float(pr.mean()), "pos_rate": float(yv.mean()),
               "matches_per_s1": float(n.mean()), "empty_share": float((n == 0).mean())}


def paired(a: pd.Series, b: pd.Series) -> dict:
    """b - a on the same S1s: mean, normal 95% CI, S1s better / worse."""
    d = (b - a.reindex(b.index)).to_numpy()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"delta": float(d.mean()), "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
            "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "n": len(d)}


def half_check(half: str, Xtr, Xdev, y_tr, y_dev, params, rounds, ctry, scales=(1.0, 1.25)) -> tuple[dict, pd.DataFrame]:
    """Train with/without the other dev half, score on this half (paired). Returns the summary + per-S1 scores."""
    A, B = halves(y_dev)
    H, O = (A, B) if half == "A" else (B, A)
    XH = Xdev[Xdev.s1_id.isin(H)].reset_index(drop=True)
    Xadd = pd.concat([Xtr, Xdev[Xdev.s1_id.isin(O)]], ignore_index=True)
    yH = {k: y_dev[k] for k in y_dev if k in H}
    y_add = {**y_tr, **{k: y_dev[k] for k in y_dev if k in O}}
    fH = design(XH)
    arms, probs = {}, {}
    for seed in (0, 1):
        for name, X, y in (("base", Xtr, y_tr), ("add", Xadd, y_add)):
            r = fit_oof(X, y, params, rounds, seed)
            arms[f"{name}_g{seed}"] = r
            probs[f"{name}_g{seed}"] = predict(r["models"], fH)
            log(f"half {half}: {name}_g{seed} fit on {X.s1_id.nunique():,} S1, best_iter {r['best_iter']}")
    for name, X, src, sc in [("full_base", Xtr, "base_g0", (1.0,))] + [("full_add", Xadd, "add_g0", scales)]:
        for s in sc:
            n = int(round(np.mean(arms[src]["best_iter"]) * s))
            m = fit_full(X, params, n)
            key = f"{name}_x{s:.2f}"
            arms[key] = {**{k: arms[src][k] for k in ("rule", "t", "oof_f05")}, "rounds": n}
            probs[key] = m.predict(fH[m.feature_name()])
            log(f"half {half}: {key} {n} rounds")
    scores, out = {}, {}
    for k, pr in probs.items():
        s, summ = evaluate(XH[["s1_id", "cand_id", "y"]].assign(prob=pr), yH, arms[k]["rule"], arms[k]["t"], ctry)
        scores[k] = s
        out[k] = {**summ, "oof_f05": arms[k]["oof_f05"], "best_iter": arms[k].get("best_iter"),
                  "rounds": arms[k].get("rounds"), "oof_minus_dev": arms[k]["oof_f05"] - summ["dev_f05"]}
    cmp = {"seed_noise (base_g1 - base_g0)": paired(scores["base_g0"], scores["base_g1"]),
           "add_g0 - base_g0": paired(scores["base_g0"], scores["add_g0"]),
           "add_g1 - base_g1": paired(scores["base_g1"], scores["add_g1"]),
           "full_base_x1.00 - base_g0": paired(scores["base_g0"], scores["full_base_x1.00"])}
    for s in scales:
        cmp[f"full_add_x{s:.2f} - base_g0"] = paired(scores["base_g0"], scores[f"full_add_x{s:.2f}"])
        cmp[f"full_add_x{s:.2f} - add_g0"] = paired(scores["add_g0"], scores[f"full_add_x{s:.2f}"])
    return {"mode": f"half{half}", "n_eval": len(yH), "n_fit_base": len(y_tr), "n_fit_add": len(y_add),
            "arms": out, "paired": cmp}, pd.DataFrame(scores)


def final_check(Xtr, Xdev, y_tr, y_dev, params, rounds, Xt, tctry, train_countries: set,
                unseen_t: float = 0.85) -> dict:
    """Final-mode arms (dev folded into fit) and their test decision drift vs the base arm, per country. Countries
    absent from train are decided at the team's unseen-country threshold, as in er_fullpass --unseen-threshold."""
    Xall = pd.concat([Xtr, Xdev], ignore_index=True)
    y_all = {**y_tr, **y_dev}
    ft = design(Xt)
    arms, probs = {}, {}
    for name, X, y, seed in (("base_g0", Xtr, y_tr, 0), ("base_g1", Xtr, y_tr, 1), ("add_g0", Xall, y_all, 0)):
        r = fit_oof(X, y, params, rounds, seed)
        arms[name] = r
        probs[name] = predict(r["models"], ft)
        log(f"final: {name} fit on {X.s1_id.nunique():,} S1, best_iter {r['best_iter']}, rule {r['rule']} t {r['t']}")
    n = int(round(np.mean(arms["add_g0"]["best_iter"])))
    m = fit_full(Xall, params, n)
    arms["full_add_x1.00"] = {**{k: arms["add_g0"][k] for k in ("rule", "t", "oof_f05")}, "rounds": n}
    probs["full_add_x1.00"] = m.predict(ft[m.feature_name()])
    s1 = pd.Index(Xt.s1_id.unique())
    c = tctry.reindex(s1).to_numpy()
    is_unseen = ~pd.Series(c, index=s1).isin(train_countries)
    dec = {}
    for k, pr in probs.items():
        p = Xt[["s1_id", "cand_id"]].assign(prob=pr)
        d = apply_decision(p, arms[k]["rule"], arms[k]["t"])
        du = decide(p[is_unseen.reindex(p.s1_id).to_numpy()], unseen_t, assign=True)
        dec[k] = ({x: frozenset(d.get(x, ())) for x in s1}, {x: frozenset(du.get(x, ())) for x in s1})
    out = {"mode": "final", "arms": {k: {kk: v for kk, v in a.items() if kk != "models"} for k, a in arms.items()},
           "test_s1": len(s1), "drift_vs_base_g0": {}}
    for k in ("base_g1", "add_g0", "full_add_x1.00"):
        row = {}
        for cc in sorted(set(c)):
            ids = s1[c == cc]
            j = 0 if cc in train_countries else 1
            a, b = dec["base_g0"][j], dec[k][j]
            ch = np.mean([a[x] != b[x] for x in ids])
            ma, mb = np.mean([len(a[x]) for x in ids]), np.mean([len(b[x]) for x in ids])
            row[cc] = {"s1_changed": float(ch), "matches_base": float(ma), "matches_arm": float(mb),
                       "matches_rel_change": float(mb / ma - 1), "empty_base": float(np.mean([not a[x] for x in ids])),
                       "empty_arm": float(np.mean([not b[x] for x in ids]))}
        pb, pk = probs["base_g0"], probs[k]
        row["pairs_mean_prob"] = [float(pb.mean()), float(pk.mean())]
        out["drift_vs_base_g0"][k] = row
    return out
