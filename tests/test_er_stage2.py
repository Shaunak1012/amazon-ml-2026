"""Cluster (second-stage) features on a toy case with a hard true match that resembles the confident anchor."""
import numpy as np
import pandas as pd
import pytest

from src.er_stage2 import cluster_features


def test_cluster_features_toy():
    right = pd.DataFrame({
        "entity_id": ["S2-a", "S3-b", "S2-x", "S3-y"],
        "name_core": ["acme foods", "acme food", "zenith", "acme foods"],
        "addr_norm": ["12 main st", "12 main street", "9 elm rd", "12 main st"],
    }).set_index("entity_id")
    e = np.array([[1, 0], [0.9, 0.1], [0, 1], [1, 0]], np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    P = pd.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-1", "S1-2"],
                      "cand_id": ["S2-a", "S3-b", "S2-x", "S3-y"],
                      "prob": [0.95, 0.40, 0.30, 0.20]})
    F = cluster_features(P, right, {"both": e.astype(np.float16)})
    assert len(F) == 4 and (F.dtypes == np.float32).all()
    # S3-b (hard true) resembles anchor S2-a far more than distractor S2-x does
    assert F.sib_cos_both_max[1] > F.sib_cos_both_max[2] + 0.5
    assert F.sib_name_tset_max[1] > F.sib_name_tset_max[2]
    assert np.isnan(F.sib_cos_both_max[0])            # the anchor itself has no other anchor
    assert F.s2_n_conf50_s1[0] == 1 and F.s2_rank_in_s1[0] == 1
    assert F.s2_other_s1_best[3] == 0.0               # S3-y is only a candidate of S1-2


def test_stage3_features_blocks_match_single_pass():
    """S1-block computation equals one pass, keeps row order, and drops the cross-S1 competition columns."""
    from types import SimpleNamespace

    from src.er_fullpass import stage3_features

    right = pd.DataFrame({
        "entity_id": ["S2-a", "S3-b", "S2-x", "S3-y", "S2-z"],
        "name_core": ["acme foods", "acme food", "zenith", "acme foods", "zen"],
        "addr_norm": ["12 main st", "12 main street", "9 elm rd", "12 main st", "9 elm"],
    }).set_index("entity_id")
    e = np.random.default_rng(0).random((5, 4)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    split = SimpleNamespace(right=right, embp={"both": e}, views=("both",))
    # S1-2's rows interleaved with S1-1's: blocks must still keep each S1's candidates together
    P = pd.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-1", "S1-1", "S1-2"],
                      "cand_id": ["S2-a", "S3-y", "S3-b", "S2-x", "S2-z"],
                      "prob": [0.95, 0.9, 0.40, 0.30, 0.6]})
    one = stage3_features(split, P)
    many = stage3_features(split, P, block=1)
    pd.testing.assert_frame_equal(one, many)
    assert len(one) == 5 and all(c.startswith("t3_") for c in one.columns)
    assert not any("other_s1" in c or "margin_vs" in c for c in one.columns)
    assert one.t3_rank_in_s1.tolist() == [1, 1, 2, 3, 2]


def test_with_ce_attaches_one_column_per_dir(tmp_path):
    from src.er_fullpass import with_ce

    chunks = [pd.DataFrame({"s1_id": ["S1-1", "S1-2"], "cand_id": ["S2-a", "S3-b"], "prob": [0.9, 0.1]})]
    for name, v in (("a", [0.8, 0.2]), ("b", [0.7, 0.3])):
        (tmp_path / name / "train_ce").mkdir(parents=True)
        chunks[0][["s1_id", "cand_id"]].assign(ce_score=v).to_parquet(tmp_path / name / "train_ce" / "000.parquet")
    out = with_ce(chunks, [str(tmp_path / "a"), str(tmp_path / "b")], "train")
    assert out[0].ce_score.tolist() == pytest.approx([0.8, 0.2]) and out[0].ce_score_2.tolist() == pytest.approx([0.7, 0.3])
    assert with_ce(chunks, "", "train") is chunks and with_ce(chunks, [], "train") is chunks


def test_stage2_frames_drop_s1_makes_record_uncontested():
    """Distractor simulation: once S1-1 is dropped, S2-a (its record) has no other S1 competing with S1-2."""
    from types import SimpleNamespace

    from src.er_fullpass import stage2_frames

    left = pd.DataFrame({"entity_id": ["S1-1", "S1-2"], "country": ["US", "US"],
                         "name_core": ["acme", "acme co"]}).set_index("entity_id")
    right = pd.DataFrame({"entity_id": ["S2-a", "S3-b"], "country": ["US", "US"], "name_core": ["acme", "zen"],
                          "addr_norm": ["1 main st", "9 elm rd"]}).set_index("entity_id")
    e = np.eye(2, dtype=np.float32)
    split = SimpleNamespace(left=left, right=right, embp={"both": e}, views=("both",))
    chunks = [pd.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-2"], "cand_id": ["S2-a", "S2-a", "S3-b"],
                            "prob": [0.9, 0.6, 0.2]})]
    full = stage2_frames(split, chunks, None)
    dropped = stage2_frames(split, chunks, None, drop_s1={"S1-1"})
    assert len(full) == 3 and len(dropped) == 2 and set(dropped.s1_id) == {"S1-2"}
    row = full[(full.s1_id == "S1-2") & (full.cand_id == "S2-a")].iloc[0]
    assert row.s2_other_s1_best == pytest.approx(0.9)          # S1-1 competes for S2-a
    assert dropped.s2_other_s1_best[0] == 0.0                   # ...and is gone after the drop
    assert dropped.rar_s1_same_name_s1[0] == 1.0 and len(chunks[0]) == 3   # input chunks untouched


def test_competition_features_on_other_column():
    """A record's best OTHER S1 by a similarity column: the S1 with the better name gets a positive margin."""
    from src.er_stage2 import competition_features

    P = pd.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-1"], "cand_id": ["S2-a", "S2-a", "S3-b"],
                      "prob": [0.1, 0.1, 0.9], "cos_name": [0.99, 0.90, 0.5]})
    f = competition_features(P, "cos_name")
    assert set(f) == {"comp_cos_name_other_best", "comp_cos_name_margin"}
    assert f["comp_cos_name_margin"].tolist() == pytest.approx([0.09, -0.09, 0.5])
    assert set(competition_features(P)) == {"s2_other_s1_best", "s2_margin_vs_other_s1"}   # prob keeps old names


def test_cached_frame_builds_once_and_rejects_other_settings(tmp_path):
    from types import SimpleNamespace

    from src.er_fullpass import cached_frame

    a = SimpleNamespace(frames=str(tmp_path / "f"), exp="E1", ce_dir=["c"], comp_cols=[], views=["both"],
                        fit_folds=[0], train_s1=10)
    calls = []

    def build():
        calls.append(1)
        return pd.DataFrame({"s1_id": ["S1-1"], "x": [0.5]})

    first = cached_frame(a, "train", build)
    second = cached_frame(a, "train", build)
    assert len(calls) == 1 and second.equals(first)
    a.comp_cols = ["name_ratio"]
    with pytest.raises(ValueError):
        cached_frame(a, "train", build)
    a.frames = ""
    assert cached_frame(a, "train", build).equals(first) and len(calls) == 2    # no cache dir: always build


def test_frames2_repopulate_matches_rebuilt_frame():
    """Recomputing competition + rarity on a cached frame equals building the frame with the S1 already dropped."""
    from types import SimpleNamespace

    from src.er_frames2 import repopulate
    from src.er_fullpass import stage2_frames

    left = pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"], "country": ["US"] * 3,
                         "name_core": ["acme", "acme co", "zen"]}).set_index("entity_id")
    right = pd.DataFrame({"entity_id": ["S2-a", "S3-b", "S2-c"], "country": ["US"] * 3,
                          "name_core": ["acme", "zen", "acme"], "addr_norm": ["1 main", "9 elm", "1 main"]}).set_index("entity_id")
    split = SimpleNamespace(left=left, right=right, embp={"both": np.eye(3, dtype=np.float32)}, views=("both",))
    chunks = [pd.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-2", "S1-3", "S1-1"],
                            "cand_id": ["S2-a", "S2-a", "S3-b", "S3-b", "S2-c"],
                            "prob": [0.9, 0.6, 0.2, 0.7, 0.4], "cos_name": [0.9, 0.8, 0.1, 0.95, 0.3]})]
    full = stage2_frames(split, chunks, None, ("cos_name",))
    want = stage2_frames(split, chunks, None, ("cos_name",), drop_s1={"S1-1"}).sort_values(["s1_id", "cand_id"])
    got = repopulate(full[full.s1_id != "S1-1"].reset_index(drop=True), chunks[0], split, {"S1-1"}, ("cos_name",))
    got = got.sort_values(["s1_id", "cand_id"])
    cols = [c for c in want.columns if c.startswith(("s2_other", "s2_margin", "comp_", "rar_"))]
    assert len(cols) == 10
    pd.testing.assert_frame_equal(got[cols].reset_index(drop=True), want[cols].reset_index(drop=True))


def test_cand_mask_topk_and_floor():
    from src.er_frames2 import cand_mask

    df = pd.DataFrame({"s1_id": ["a", "a", "a", "b", "b"], "prob": [0.9, 0.05, 0.4, 0.001, 0.3]})
    assert cand_mask(df, 2, 0.0).tolist() == [True, False, True, True, True]
    assert cand_mask(df, 15, 0.01).tolist() == [True, True, True, False, True]
    assert cand_mask(df, 1, 0.5).tolist() == [True, False, False, False, False]   # b ends with no candidates
