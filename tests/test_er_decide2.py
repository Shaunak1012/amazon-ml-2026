"""Exact expected-F_beta prefix equals brute-force enumeration over all label vectors."""
import itertools

import numpy as np
import pandas as pd
import pytest

from src.er_decide2 import decide_expected_f_exact, expected_f_prefix_exact


def brute(p, beta=0.5):
    b2, m = beta * beta, len(p)
    vals = []
    for k in range(m + 1):
        v = 0.0
        for ys in itertools.product([0, 1], repeat=m):
            w = np.prod([q if t else 1 - q for q, t in zip(p, ys)])
            tp, n = sum(ys[:k]), sum(ys)
            f = (1.0 if n == 0 else 0.0) if k == 0 else (1 + b2) * tp / (b2 * n + k)
            v += w * f
        vals.append(v)
    return int(np.argmax(vals)), max(vals)


@pytest.mark.parametrize("seed", range(20))
def test_exact_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    p = np.sort(rng.random(rng.integers(1, 8)))[::-1]
    k, v = expected_f_prefix_exact(p)
    kb, vb = brute(p)
    assert v == pytest.approx(vb, abs=1e-12) and k == kb


def test_decide_exact_frame():
    p = pd.DataFrame({"s1_id": ["a", "a", "b", "c"], "cand_id": ["x", "y", "z", "x"], "prob": [0.95, 0.9, 0.05, 0.6]})
    out = decide_expected_f_exact(p)
    assert out == {"a": {"x", "y"}}          # x goes to a (higher prob), b is a confident singleton
