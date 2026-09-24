# Submission ledger

Daily limit: **5** (15 total over 25–27 Sep) · Reset: assumed 00:00 IST · Window closes **27 Sep 23:59 IST** · Reserve ≥ 2 for 27 Sep.

Gate before each submission (see CLAUDE.md → Submission discipline):
1. CV improved meaningfully over best submitted run (> fold std)? 2. Remaining today > reserve?
3. Code + config committed? 4. Both validators pass (`src.er_submission` + organisers' `validate_submission.py`)? → `python scripts/tag_submission.py ...`

| # | timestamp (IST) | experiment id | git tag | CV | public LB | remaining today | file |
|---|---|---|---|---|---|---|---|
| 1 | 2026-09-25 00:51 IST | P0-probe-empty | `sub-01` | n/a (probe; train singleton rate 0.0558) |  | 4 | sub01_empty |
