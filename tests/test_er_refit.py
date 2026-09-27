"""RF01 refit check on a toy stage-2 frame (no data, no GPU)."""
import numpy as np
import pandas as pd


def toy(n_s1: int, seed: int, prefix: str):
    rng = np.random.default_rng(seed)
    rows, y = [], {}
    for i in range(n_s1):
        s = f"{prefix}{i}"
        y[s] = set()
        for j in range(3):
            c = f"{s}_c{j}"
            lab = int(rng.random() < 0.4)
            if lab:
                y[s].add(c)
            rows.append((s, c, lab, float(np.clip(0.6 * lab + 0.4 * rng.random(), 0, 1)), rng.normal(lab, 1.0)))
    X = pd.DataFrame(rows, columns=["s1_id", "cand_id", "y", "prob", "f1"])
    return X, y


def test_halves_partition_and_deterministic():
    from src.er_refit import halves
    ids = [f"d{i}" for i in range(101)]
    a, b = halves(ids)
    assert a | b == set(ids) and not a & b and len(a) == 50
    assert halves(list(reversed(ids))) == (a, b)


def test_paired_delta_matches_means():
    from src.er_refit import paired
    a = pd.Series([1.0, 0.5, 1.0, 0.0], index=list("wxyz"))
    b = pd.Series([1.0, 1.0, 0.5, 0.0], index=list("zyxw"))
    r = paired(a, b)
    assert abs(r["delta"] - (b.mean() - a.mean())) < 1e-12
    assert r["n_better"] == 1 and r["n_worse"] == 1 and r["n"] == 4     # w -1, x 0, y 0, z +1


def test_half_check_runs_and_keeps_eval_rows_out_of_training():
    from src import er_refit
    Xtr, y_tr = toy(400, 0, "f")
    Xdev, y_dev = toy(200, 1, "d")
    ctry = pd.Series("US", index=list(y_tr) + list(y_dev))
    params = {"num_threads": 1, "min_data_in_leaf": 5, "num_leaves": 7}
    res, scores = er_refit.half_check("A", Xtr, Xdev, y_tr, y_dev, params, 30, ctry)
    A, B = er_refit.halves(y_dev)
    assert set(scores.index) == A and res["n_eval"] == len(A)
    assert res["n_fit_add"] == len(y_tr) + len(B)
    for k, v in res["arms"].items():
        assert 0.0 <= v["dev_f05"] <= 1.0, k
        assert abs(scores[k].mean() - v["dev_f05"]) < 1e-12
    assert "add_g0 - base_g0" in res["paired"] and "full_add_x1.25 - add_g0" in res["paired"]


def test_final_check_drift_per_country():
    from src import er_refit
    Xtr, y_tr = toy(300, 0, "f")
    Xdev, y_dev = toy(100, 1, "d")
    Xt, _ = toy(200, 2, "t")
    tctry = pd.Series(["US"] * 100 + ["France"] * 100, index=[f"t{i}" for i in range(200)])
    params = {"num_threads": 1, "min_data_in_leaf": 5, "num_leaves": 7}
    res = er_refit.final_check(Xtr, Xdev, y_tr, y_dev, params, 30, Xt.drop(columns=["y"]), tctry, {"US", "India"})
    d = res["drift_vs_base_g0"]
    assert set(d) == {"base_g1", "add_g0", "full_add_x1.00"}
    assert set(d["add_g0"]) >= {"US", "France"}
    assert 0.0 <= d["add_g0"]["France"]["s1_changed"] <= 1.0
