from scripts.e041_oracle_ceiling import (
    analyze,
    oracle_f05,
    summarize,
)


def test_oracle_formula():
    assert oracle_f05(0.0) == 0.0
    assert abs(oracle_f05(1.0) - 1.0) < 1e-12

    expected = (
        1.25 * (2 / 3)
        / (0.25 + (2 / 3))
    )

    assert abs(
        oracle_f05(2 / 3) - expected
    ) < 1e-12


def test_true_empty_gets_perfect_oracle():
    dev = [
        "S1-empty",
        "S1-match",
    ]

    gt = {
        "S1-empty": set(),
        "S1-match": {
            "S2-a",
            "S3-a",
            "S3-b",
        },
    }

    hits = {
        "S1-empty": 0,
        "S1-match": 2,
    }

    df = analyze(
        dev,
        gt,
        hits,
        {},
    )

    empty = df[
        df.source1_entity_id
        == "S1-empty"
    ].iloc[0]

    matched = df[
        df.source1_entity_id
        == "S1-match"
    ].iloc[0]

    assert bool(empty.is_true_empty)
    assert empty.oracle_f05 == 1.0

    assert matched.n_hit == 2
    assert matched.n_true == 3
    assert not bool(matched.fully_missed)

    summary = summarize(df)

    assert summary["n_dev_s1"] == 2
    assert summary["n_true_empty"] == 1
