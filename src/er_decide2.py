"""Exact expected-F_beta decisions (replaces the 256-sample Monte-Carlo in er_decide.decide_expected_f).

For one S1 with candidates sorted by p (independent Bernoulli labels), predicting the top-k gives
    F = (1 + b2) * TP / (b2 * (TP + FN) + k),  TP ~ PoissonBinomial(p[:k]), FN ~ PoissonBinomial(p[k:])
and the empty prediction scores P(TP + FN == 0). The exact expectation needs the two count distributions, built by
convolution in O(m^2) per prefix; m <= 20 candidates, so the whole dev set takes seconds with numpy.
True matches that were never retrieved are not modelled (same as the Monte-Carlo version).

    python -m src.er_decide2 --dev runs/shared/E015/dev_stage2_E019.parquet     # compares decision variants on dev
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.er_decide import assign_records


def _pb(p: np.ndarray) -> np.ndarray:
    """Poisson-binomial pmf of the number of successes."""
    d = np.zeros(len(p) + 1)
    d[0] = 1.0
    for i, q in enumerate(p):
        d[1:i + 2] = d[1:i + 2] * (1 - q) + d[:i + 1] * q
        d[0] *= 1 - q
    return d


def expected_f_prefix_exact(probs: np.ndarray, beta: float = 0.5) -> tuple[int, float]:
    """(best prefix length, its expected F_beta); 0 = empty prediction."""
    p = np.asarray(probs, dtype=np.float64)
    m = len(p)
    if m == 0:
        return 0, 1.0
    b2 = beta * beta
    best_k, best_v = 0, float(np.prod(1 - p))
    for k in range(1, m + 1):
        tp, fn = _pb(p[:k]), _pb(p[k:])
        t = np.arange(k + 1)[:, None]
        f = np.arange(m - k + 1)[None, :]
        v = float((tp[:, None] * fn[None, :] * ((1 + b2) * t / (b2 * (t + f) + k))).sum())
        if v > best_v:
            best_k, best_v = k, v
    return best_k, best_v


def decide_expected_f_exact(p: pd.DataFrame, beta: float = 0.5, assign: bool = True, max_cands: int = 20,
                            min_prob: float = 0.01) -> dict[str, set[str]]:
    """Drop-in for er_decide.decide_expected_f with the exact expectation."""
    q = assign_records(p) if assign else p
    q = q[q.prob >= min_prob].sort_values(["s1_id", "prob"], ascending=[True, False], kind="stable")
    out: dict[str, set[str]] = {}
    ids, probs, cands = q.s1_id.to_numpy(), q.prob.to_numpy(), q.cand_id.to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    for s, e in zip(starts, ends):
        e = min(e, s + max_cands)
        k, _ = expected_f_prefix_exact(probs[s:e], beta)
        if k:
            out[ids[s]] = set(cands[s:s + k])
    return out


def main() -> None:
    from sklearn.isotonic import IsotonicRegression

    from src.er_data import cache_dir
    from src.er_decide import decide_expected_f
    from src.metrics import er_fbeta_macro

    ap = argparse.ArgumentParser(prog="python -m src.er_decide2")
    ap.add_argument("--dev", required=True, help="dev_stage2_<tag>.parquet (s1_id, cand_id, y, prob, ...)")
    ap.add_argument("--gt", default="", help="train_gt_pairs.parquet (default: data/cache)")
    a = ap.parse_args()
    d = pd.read_parquet(a.dev)
    gt = pd.read_parquet(a.gt or cache_dir() / "train_gt_pairs.parquet")
    ids = d.s1_id.unique()
    g = gt[gt.s1_id.isin(set(ids))]
    y = {s: set() for s in ids}
    for s, c in zip(g.s1_id.to_numpy(), g.cand_id.to_numpy()):
        y[s].add(c)
    p = d[["s1_id", "cand_id", "prob"]]
    res = {"mc256 (current)": er_fbeta_macro(y, decide_expected_f(p)),
           "exact": er_fbeta_macro(y, decide_expected_f_exact(p))}
    # isotonic calibration, cross-fitted over two halves of dev S1 (never calibrates on its own rows)
    half = pd.Series(np.random.default_rng(0).integers(0, 2, len(ids)), index=ids).reindex(d.s1_id).to_numpy()
    cal = np.empty(len(d))
    for h in (0, 1):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(d.prob[half != h], d.y[half != h])
        cal[half == h] = iso.predict(d.prob[half == h])
    pc = p.assign(prob=cal)
    res["exact + isotonic (2-fold on dev)"] = er_fbeta_macro(y, decide_expected_f_exact(pc))
    for name, v in res.items():
        print(f"{name:36s} {v:.5f}")


if __name__ == "__main__":
    main()
