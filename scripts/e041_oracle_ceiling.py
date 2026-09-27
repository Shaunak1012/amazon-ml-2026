#!/home/ec2-user/anaconda3/envs/pytorch/bin/python3.10
"""
E041: entity-level oracle upper-bound analysis for candidate retrieval.

For each fixed-dev S1:
  matched S1 (n_true > 0):
      recall = retrieved GT matches / total GT matches
      oracle F0.5 assumes perfect precision (=1)
  true-empty S1:
      oracle F0.5 = 1.0

The headline score is macro-averaged over S1 entities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.dataset as ds


S1_CANDIDATE_COLS = (
    "source1_entity_id",
    "s1_entity_id",
    "query_entity_id",
    "entity_id",
)
CANDIDATE_ID_COLS = (
    "candidate_entity_id",
    "matched_entity_id",
    "candidate_id",
    "entity_id",
)
GT_S1_COLS = (
    "source1_entity_id",
    "s1_entity_id",
    "entity_id",
)
GT_MATCH_COLS = (
    "matched_entity_ids",
    "matched_entity_id",
    "candidates",
)


def pick_col(
    columns: Iterable[str],
    preferred: Iterable[str],
    explicit: str | None,
    kind: str,
) -> str:
    cols = set(columns)
    if explicit:
        if explicit not in cols:
            raise ValueError(
                f"{kind} column {explicit!r} not found. "
                f"Available: {sorted(cols)}"
            )
        return explicit

    for col in preferred:
        if col in cols:
            return col

    raise ValueError(
        f"Could not auto-detect {kind} column. "
        f"Available: {sorted(cols)}"
    )


def load_gt(
    gt_path: Path,
    explicit_s1_col: str | None,
) -> dict[str, set[str]]:
    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s1_col = pick_col(
        gt.columns,
        GT_S1_COLS,
        explicit_s1_col,
        "GT S1",
    )
    match_col = pick_col(
        gt.columns,
        GT_MATCH_COLS,
        None,
        "GT match",
    )

    out: dict[str, set[str]] = {}

    for s1, raw in zip(
        gt[s1_col].astype(str),
        gt[match_col].astype(str),
    ):
        raw = raw.strip()
        matches = (
            {x.strip() for x in raw.split(",") if x.strip()}
            if raw
            else set()
        )

        if s1 in out:
            raise ValueError(f"Duplicate GT row for {s1}")

        out[s1] = matches

    return out


def load_dev_ids(
    path: Path,
    explicit_col: str | None,
) -> list[str]:
    df = pd.read_parquet(path)

    col = explicit_col
    if col is None:
        for candidate in (
            "source1_entity_id",
            "entity_id",
            "s1_entity_id",
        ):
            if candidate in df.columns:
                col = candidate
                break

    if col is None or col not in df.columns:
        raise ValueError(
            f"Could not identify dev S1 ID column in {path}; "
            f"columns={list(df.columns)}"
        )

    ids = df[col].astype(str).tolist()

    if len(ids) != len(set(ids)):
        raise ValueError("Dev S1 ID file contains duplicate IDs")

    return ids


def load_country_map(
    path: Path | None,
) -> dict[str, str]:
    if path is None:
        return {}

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )

    id_col = next(
        (
            c
            for c in (
                "entity_id",
                "source1_entity_id",
                "s1_entity_id",
            )
            if c in df.columns
        ),
        None,
    )
    country_col = (
        "country"
        if "country" in df.columns
        else None
    )

    if id_col is None or country_col is None:
        raise ValueError(
            "Country source needs an S1 ID and country column"
        )

    return dict(
        zip(
            df[id_col].astype(str),
            df[country_col].astype(str),
        )
    )


def discover_parquet(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = sorted(path.rglob("*.parquet"))

    if not candidates:
        raise FileNotFoundError(
            f"No parquet files under {path}"
        )

    ranked = sorted(
        candidates,
        key=lambda p: (
            0 if "candidate" in p.name.lower() else 1,
            0 if "union" in p.name.lower() else 1,
            -p.stat().st_size,
        ),
    )

    return ranked[0]


def collect_hits(
    candidate_path: Path,
    dev_set: set[str],
    gt_map: dict[str, set[str]],
    explicit_s1_col: str | None,
    explicit_candidate_col: str | None,
    batch_size: int,
) -> tuple[
    dict[str, int],
    int,
    str,
    str,
]:
    dataset = ds.dataset(
        candidate_path,
        format="parquet",
    )

    schema_cols = dataset.schema.names

    s1_col = pick_col(
        schema_cols,
        S1_CANDIDATE_COLS,
        explicit_s1_col,
        "candidate S1",
    )

    cand_col = pick_col(
        schema_cols,
        CANDIDATE_ID_COLS,
        explicit_candidate_col,
        "candidate ID",
    )

    hit_counts = {
        s1: 0
        for s1 in dev_set
    }

    seen_pairs: set[tuple[str, str]] = set()
    total_rows = 0

    scanner = dataset.scanner(
        columns=[s1_col, cand_col],
        batch_size=batch_size,
    )

    for batch in scanner.to_batches():
        df = batch.to_pandas()

        total_rows += len(df)

        df[s1_col] = df[s1_col].astype(str)
        df[cand_col] = df[cand_col].astype(str)

        df = df[
            df[s1_col].isin(dev_set)
        ]

        if df.empty:
            continue

        for s1, cand in df[
            [s1_col, cand_col]
        ].itertuples(index=False):
            if not cand or s1 not in gt_map:
                continue

            key = (s1, cand)

            if key in seen_pairs:
                continue

            seen_pairs.add(key)

            if cand in gt_map[s1]:
                hit_counts[s1] += 1

    return (
        hit_counts,
        total_rows,
        s1_col,
        cand_col,
    )


def oracle_f05(recall: float) -> float:
    if recall <= 0:
        return 0.0

    return (
        1.25
        * recall
        / (0.25 + recall)
    )


def analyze(
    dev_ids: list[str],
    gt_map: dict[str, set[str]],
    hit_counts: dict[str, int],
    country_map: dict[str, str],
) -> pd.DataFrame:
    rows = []

    for s1 in dev_ids:
        true_set = gt_map.get(
            s1,
            set(),
        )

        n_true = len(true_set)

        if n_true == 0:
            score = 1.0
            n_hit = None
            recall = None
            fully_missed = False

        else:
            n_hit = min(
                hit_counts.get(s1, 0),
                n_true,
            )

            recall = n_hit / n_true
            score = oracle_f05(recall)
            fully_missed = (
                n_hit == 0
            )

        rows.append(
            {
                "source1_entity_id": s1,
                "country": country_map.get(
                    s1,
                    "",
                ),
                "n_true": n_true,
                "n_hit": n_hit,
                "retrieval_recall": recall,
                "fully_missed": fully_missed,
                "is_true_empty": (
                    n_true == 0
                ),
                "oracle_f05": score,
            }
        )

    return pd.DataFrame(rows)


def summarize(
    df: pd.DataFrame,
) -> dict:
    matched = df[
        ~df["is_true_empty"]
    ]

    return {
        "n_dev_s1": int(len(df)),
        "n_true_empty": int(
            df["is_true_empty"].sum()
        ),
        "n_matched_entities": int(
            len(matched)
        ),
        "oracle_f05_macro": float(
            df["oracle_f05"].mean()
        ),
        "matched_entity_oracle_f05": (
            float(
                matched["oracle_f05"].mean()
            )
            if len(matched)
            else None
        ),
        "fully_missed_entities": int(
            df["fully_missed"].sum()
        ),
        "fully_missed_rate_among_matched": (
            float(
                matched["fully_missed"].mean()
            )
            if len(matched)
            else None
        ),
        "mean_retrieval_recall_among_matched": (
            float(
                matched[
                    "retrieval_recall"
                ].mean()
            )
            if len(matched)
            else None
        ),
        "retrieval_recall_p10": (
            float(
                matched[
                    "retrieval_recall"
                ].quantile(0.10)
            )
            if len(matched)
            else None
        ),
        "retrieval_recall_p50": (
            float(
                matched[
                    "retrieval_recall"
                ].quantile(0.50)
            )
            if len(matched)
            else None
        ),
        "retrieval_recall_p90": (
            float(
                matched[
                    "retrieval_recall"
                ].quantile(0.90)
            )
            if len(matched)
            else None
        ),
        "oracle_f05_p10_matched": (
            float(
                matched[
                    "oracle_f05"
                ].quantile(0.10)
            )
            if len(matched)
            else None
        ),
        "oracle_f05_p50_matched": (
            float(
                matched[
                    "oracle_f05"
                ].quantile(0.50)
            )
            if len(matched)
            else None
        ),
        "oracle_f05_p90_matched": (
            float(
                matched[
                    "oracle_f05"
                ].quantile(0.90)
            )
            if len(matched)
            else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--gt",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--dev-ids",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--candidates",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--country-source",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--gt-s1-col",
        default=None,
    )
    ap.add_argument(
        "--candidate-s1-col",
        default=None,
    )
    ap.add_argument(
        "--candidate-id-col",
        default=None,
    )
    ap.add_argument(
        "--dev-id-col",
        default=None,
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=250_000,
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    candidate_path = discover_parquet(
        args.candidates
    )

    gt_map = load_gt(
        args.gt,
        args.gt_s1_col,
    )

    dev_ids = load_dev_ids(
        args.dev_ids,
        args.dev_id_col,
    )

    dev_set = set(dev_ids)

    country_map = load_country_map(
        args.country_source
    )

    missing_gt = [
        s
        for s in dev_ids
        if s not in gt_map
    ]

    if missing_gt:
        raise ValueError(
            f"{len(missing_gt)} dev S1 IDs "
            f"are missing from GT; "
            f"first={missing_gt[:5]}"
        )

    (
        hit_counts,
        rows_scanned,
        cand_s1_col,
        cand_id_col,
    ) = collect_hits(
        candidate_path,
        dev_set,
        gt_map,
        args.candidate_s1_col,
        args.candidate_id_col,
        args.batch_size,
    )

    df = analyze(
        dev_ids,
        gt_map,
        hit_counts,
        country_map,
    )

    summary = summarize(df)

    by_country = {}

    if (df["country"] != "").any():
        for country, group in df[
            df["country"] != ""
        ].groupby("country"):
            matched_group = group[
                ~group["is_true_empty"]
            ]

            by_country[str(country)] = {
                "n_dev_s1": int(len(group)),
                "oracle_f05": float(
                    group[
                        "oracle_f05"
                    ].mean()
                ),
                "fully_missed_rate": float(
                    group[
                        "fully_missed"
                    ].mean()
                ),
                "mean_retrieval_recall_matched": (
                    float(
                        matched_group[
                            "retrieval_recall"
                        ].mean()
                    )
                    if len(matched_group)
                    else None
                ),
            }

    payload = {
        "candidate_path": str(
            candidate_path
        ),
        "candidate_s1_col": cand_s1_col,
        "candidate_id_col": cand_id_col,
        "rows_scanned_in_candidate_artifact": rows_scanned,
        "summary": summary,
        "by_country": by_country,
    }

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.out.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    df.to_parquet(
        args.out.with_suffix(
            ".per_s1.parquet"
        ),
        index=False,
    )

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
