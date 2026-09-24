"""Competition metrics. Implement the official formula EXACTLY on Day 1 and unit-test it.

Every metric: fn(y_true, y_pred) -> float. Registry records direction so CV loops,
heartbeat, and early stopping never get "higher vs lower is better" wrong.

    from src.metrics import get_metric
    fn, higher_is_better = get_metric("smape")

Edge cases (zeros, empty strings, NaN) are where leaderboard formulas differ from
textbook ones — copy the organisers' definition, then add a test for each edge case.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

MetricFn = Callable[..., float]
_REGISTRY: dict[str, tuple[MetricFn, bool]] = {}


def register(name: str, higher_is_better: bool):
    def deco(fn: MetricFn) -> MetricFn:
        _REGISTRY[name] = (fn, higher_is_better)
        return fn

    return deco


def get_metric(name: str) -> tuple[MetricFn, bool]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_metrics() -> list[str]:
    return sorted(_REGISTRY)


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains NaN/inf")


# ---------------------------------------------------------------- regression
@register("smape", higher_is_better=False)
def smape(y_true, y_pred) -> float:
    """SMAPE in percent, range [0, 200]. 0/0 terms count as 0 error (2025 challenge style).

    SMAPE = 100/n * sum(|F - A| / ((|A| + |F|) / 2))
    Verify zero-handling against the official definition on Day 1.
    """
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    diff = np.abs(f - a)
    ratio = np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0)
    return float(100.0 * ratio.mean())


@register("rmse", higher_is_better=False)
def rmse(y_true, y_pred) -> float:
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    return float(np.sqrt(np.mean((a - f) ** 2)))


@register("mae", higher_is_better=False)
def mae(y_true, y_pred) -> float:
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    return float(np.mean(np.abs(a - f)))


@register("rmsle", higher_is_better=False)
def rmsle(y_true, y_pred) -> float:
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    if (a < 0).any() or (f < 0).any():
        raise ValueError("rmsle requires non-negative values")
    return float(np.sqrt(np.mean((np.log1p(a) - np.log1p(f)) ** 2)))


@register("mape", higher_is_better=False)
def mape(y_true, y_pred) -> float:
    """MAPE in percent; rows with y_true == 0 are excluded (check official handling!)."""
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    m = a != 0
    return float(100.0 * np.mean(np.abs((a[m] - f[m]) / a[m])))


@register("r2", higher_is_better=True)
def r2(y_true, y_pred) -> float:
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    ss_res = np.sum((a - f) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ------------------------------------------------------------ classification
@register("accuracy", higher_is_better=True)
def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


@register("f1_macro", higher_is_better=True)
def f1_macro(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro"))


@register("f1_micro", higher_is_better=True)
def f1_micro(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="micro"))


@register("auc", higher_is_better=True)
def auc(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


@register("logloss", higher_is_better=False)
def logloss(y_true, y_prob) -> float:
    from sklearn.metrics import log_loss

    return float(log_loss(y_true, np.clip(_arr(y_prob), 1e-15, 1 - 1e-15)))


# ---------------------------------------------------------------- extraction
@register("extraction_f1", higher_is_better=True)
def extraction_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """2024-style entity-extraction F1 on exact string match; "" means "no prediction".

    TP: pred!="" & gt!="" & pred==gt     FP: pred!="" & (gt=="" or pred!=gt)
    FN: pred=="" & gt!=""                TN: pred=="" & gt==""
    """
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    tp = fp = fn = 0
    for gt, out in zip(y_true, y_pred):
        gt = "" if gt is None or (isinstance(gt, float) and np.isnan(gt)) else str(gt).strip()
        out = "" if out is None or (isinstance(out, float) and np.isnan(out)) else str(out).strip()
        if out and gt and out == gt:
            tp += 1
        elif out:
            fp += 1
        elif gt:
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * p * r / (p + r)) if p + r else 0.0


# ------------------------------------------------------------------- ranking
@register("ndcg", higher_is_better=True)
def ndcg(y_true, y_score, k: int | None = None) -> float:
    """y_true/y_score: 2D arrays (n_queries, n_items) of relevance / scores."""
    from sklearn.metrics import ndcg_score

    return float(ndcg_score(np.atleast_2d(y_true), np.atleast_2d(y_score), k=k))


@register("map_at_k", higher_is_better=True)
def map_at_k(actual: Sequence[Sequence], predicted: Sequence[Sequence], k: int = 10) -> float:
    """Mean average precision@k over queries (Kaggle definition)."""
    scores = []
    for act, pred in zip(actual, predicted):
        act = set(act)
        if not act:
            continue
        hits, s = 0, 0.0
        for i, p in enumerate(list(pred)[:k]):
            if p in act and p not in list(pred)[:i]:
                hits += 1
                s += hits / (i + 1)
        scores.append(s / min(len(act), k))
    return float(np.mean(scores)) if scores else 0.0
