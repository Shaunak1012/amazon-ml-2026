import math

import numpy as np
import pytest

from src.metrics import get_metric, list_metrics


def m(name):
    return get_metric(name)[0]


def test_registry_directions():
    assert get_metric("smape")[1] is False
    assert get_metric("f1_macro")[1] is True
    assert "extraction_f1" in list_metrics()
    with pytest.raises(KeyError):
        get_metric("nope")


def test_smape_hand_computed():
    # |110-100| / ((100+110)/2) = 10/105 ; |50-100|/((100+50)/2)=50/75
    exp = 100 * (10 / 105 + 50 / 75) / 2
    assert m("smape")([100, 100], [110, 50]) == pytest.approx(exp)


def test_smape_edges():
    assert m("smape")([0, 0], [0, 0]) == 0.0          # 0/0 -> 0
    assert m("smape")([0], [5]) == pytest.approx(200)  # max error
    assert m("smape")([5], [5]) == 0.0
    assert m("smape")([1, 2, 3], [1, 2, 3]) == 0.0
    # symmetric in the "symmetric" sense
    assert m("smape")([100], [120]) == pytest.approx(m("smape")([120], [100]))


def test_smape_rejects_nan_and_shape():
    with pytest.raises(ValueError):
        m("smape")([1, 2], [1, np.nan])
    with pytest.raises(ValueError):
        m("smape")([1, 2], [1])


def test_regression_basics():
    y, p = [1.0, 2.0, 3.0], [1.0, 2.0, 5.0]
    assert m("rmse")(y, p) == pytest.approx(math.sqrt(4 / 3))
    assert m("mae")(y, p) == pytest.approx(2 / 3)
    assert m("rmsle")([0, math.e - 1], [0, 0]) == pytest.approx(math.sqrt(0.5))
    assert m("r2")(y, y) == pytest.approx(1.0)
    assert m("mape")([0, 100], [5, 110]) == pytest.approx(10.0)  # zero target excluded


def test_classification():
    assert m("accuracy")([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
    assert m("f1_macro")([0, 1], [0, 1]) == 1.0
    assert m("auc")([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_extraction_f1_cases():
    gt = ["10 gram", "", "5 cm", "2 kg", ""]
    out = ["10 gram", "", "6 cm", "", "1 kg"]
    # TP=1, FP=2 (wrong value + pred on empty gt), FN=1, TN=1
    p, r = 1 / 3, 1 / 2
    assert m("extraction_f1")(gt, out) == pytest.approx(2 * p * r / (p + r))
    assert m("extraction_f1")(["a"], ["a"]) == 1.0
    assert m("extraction_f1")([""], [""]) == 0.0


def test_ranking():
    assert m("map_at_k")([[1, 2]], [[1, 2, 3]], k=3) == 1.0
    assert m("map_at_k")([[3]], [[1, 2, 3]], k=3) == pytest.approx(1 / 3)
    assert m("ndcg")([[3, 2, 1]], [[0.9, 0.5, 0.1]]) == pytest.approx(1.0)
