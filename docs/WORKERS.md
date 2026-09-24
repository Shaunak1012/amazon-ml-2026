# Two-worker protocol: local session + cloud session

| | **Local worker** (this PC) | **Cloud worker** (`claude --cloud`) |
|---|---|---|
| Has | real data, RTX 5080, `.env`, `oof/`, `runs/` | GitHub copy of `main`, **no data, no GPU** |
| Does | EDA, training, CV scores, blending, submissions, merging | pure code tasks from a written brief (see below) |
| Tests with | real data + `pytest` | `pytest` + synthetic data (`python -m src.er_synthetic`) |
| Owns | `main`, all docs/logs/configs, anything touching scores | only the files listed in its brief |

The local worker is the **integrator**: it writes the briefs, reviews and merges the branches, runs them on real
data, and records the results. The cloud worker never decides strategy and never edits logs.

## 1. Stage contracts (what makes parallel work safe)
One file per pipeline stage. Workers can build different stages in parallel as long as these signatures hold.
**Changing a contract is a local-worker decision and needs a DECISIONS.md entry.**

| File | Contract |
|---|---|
| `src/er_normalize.py` | pure `str -> str` / `str -> list` functions: `normalize_name`, `core_name` (legal suffix removed), `normalize_address`, `extract_numbers`, `extract_postcode`. Never branch on specific country values. |
| `src/er_blocking.py` | `generate_candidates(s1, s2, s3, cfg) -> DataFrame[s1_id, cand_id, blocker, score]` (unique `(s1_id, cand_id)`; `blocker` = which retrievers found the pair); `blocking_recall(cands, gt) -> dict` |
| `src/er_features.py` | feature groups registered as `@feature_group("name")`, each `f(pairs, s1, s2, s3) -> DataFrame` with the same row order as `pairs`; `build_features(pairs, s1, s2, s3, groups=None) -> DataFrame[s1_id, cand_id, <float32 features>]` (NaN allowed) |
| `src/er_model.py` | `train_oof(X, y, groups, cfg) -> (oof_prob, models)`; `predict(models, X) -> prob` |
| `src/er_decide.py` | `decide(pairs_with_prob, cfg) -> dict[s1_id, set[cand_id]]` (threshold, one-S1-per-record, empty allowed) |
| `src/er_pipeline.py` | CLI orchestrating the stages; writes both outputs via `src.er_submission.write_outputs` |

Common vocabulary: source frames have columns `entity_id, business_name, business_address, country` (all str);
the pair table has `s1_id, cand_id` (str), optional label `y` (0/1). Candidate source = `cand_id[:2]` (`S2`/`S3`).

Shared files that **only the local worker edits**: `CLAUDE.md`, `docs/*`, `configs/base.yaml`, `src/metrics.py`,
`src/er_submission.py`, `requirements*.txt`. If a cloud task needs a new dependency, it says so in its final
message instead of editing the requirements files.

## 2. Handoff loop
1. **Local:** commit and push `main`, add a row to `docs/TASKS.md` (status `cloud`), write the brief (template
   below), start the cloud session with it.
2. **Cloud:** branch `cloud/T<id>-<slug>` from the latest `main`; edit **only** the files in the brief plus their
   tests; meet the done criteria; push the branch; reply with a 5-line summary (what, files, how to run, open
   questions, new dependencies).
3. **Local:** `git fetch && git switch cloud/T<id>-...`; run `pytest`, then run it on the **real data** and record
   any score in EXPERIMENTS.md; merge into `main` with `git merge --no-ff` (keeps the history readable); push;
   ask the user before deleting the merged branch (CLAUDE.md: no branch deletion without asking).
   Set the task to `done`.
4. Never run two workers on the **same file** at the same time. If a brief must touch a file the local worker is
   editing, finish and push the local change first.

## 3. Done criteria (every task, both workers)
- `python -m pytest` green, **including new tests** for the new code (use `src.er_synthetic` fixtures; no real data).
- `ruff check .` clean. Functions have short docstrings (the organisers review the code).
- Runs end to end on synthetic data: `python -m src.er_synthetic --out data_synth` then the relevant CLI.
- No data, outputs, weights or secrets committed. Conventional Commit messages, **no AI/Claude mentions or
  co-author trailers** (see CLAUDE.md).
- Honour the rules: MIT/Apache models ≤ 8B, **no external data or network lookups inside the pipeline**, no
  hard-coded country lists (France is unseen in train).

## 4. Brief template (paste into the cloud session)
```
Task T<id>: <one-line goal>
Repo: Shaunak1012/amazon-ml-2026, start from latest main, branch cloud/T<id>-<slug>.
Read first: CLAUDE.md, docs/WORKERS.md (contracts + done criteria), docs/COMPETITION.md (rules).
Edit only: <files>  (+ tests/test_<...>.py)
Contract: <exact function signatures / columns expected>
Context: <what the data looks like: paste a few anonymised rows or EDA facts; the cloud worker cannot see data/>
Acceptance: <specific tests / behaviours, e.g. "core_name('ACME Pvt. Ltd.') == 'acme'">
Out of scope: <what not to touch>
When done: push the branch and reply with a summary (what, files, how to run, questions, new deps).
```

## 5. Good cloud tasks vs. local-only
- **Cloud:** normalisation rules and their tests; a new feature group; a new blocking retriever (tested for recall
  on synthetic data); refactors; the methodology-doc draft from DECISIONS/EXPERIMENTS; README/REPRODUCE polish.
- **Local only:** anything that needs a score, threshold tuning, choosing K, EDA, training, blending, submissions.
