"""Cluster (second-stage) features on a toy case with a hard true match that resembles the confident anchor."""
import numpy as np
import pandas as pd

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
