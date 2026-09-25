# Submission ledger

Daily limit: **5** (15 total over 25–27 Sep) · Reset: assumed 00:00 IST · Window closes **27 Sep 23:59 IST** · Reserve ≥ 2 for 27 Sep.

Gate before each submission (see CLAUDE.md → Submission discipline):
1. CV improved meaningfully over best submitted run (> fold std)? 2. Remaining today > reserve?
3. Code + config committed? 4. Both validators pass (`src.er_submission` + organisers' `validate_submission.py`)? → `python scripts/tag_submission.py ...`

| # | timestamp (IST) | experiment id | git tag | CV | public LB | remaining today | file |
|---|---|---|---|---|---|---|---|
| 1 | 2026-09-25 00:51 IST | P0-probe-empty | `sub-01` | n/a (probe; train singleton rate 0.0558) |  | 4 | sub01_empty |
| 2 | 2026-09-25 04:58 IST | E004-predict-sub02 | `sub-02` | 0.9476 dev / 0.9487 OOF | **FAILED at portal evaluation** (file re-verified: passes organiser validator --check-ids + independent checker) | ? | sub02_E004 |
| 3 | 2026-09-25 17:07 IST | E009-stage2-noce | `sub-03` | 0.9643 dev / 0.9618 OOF |  | ? | sub03_E009 |
| 4 | 2026-09-25 17:38 IST | probe-france-empty-of-sub03 | `sub-04` | probe (sub-03 with france emptied) |  | ? | probe_france_sub03 |
| 5 | 2026-09-25 17:38 IST | probe-india-empty-of-sub03 | `sub-05` | probe (sub-03 with india emptied) |  | ? | probe_india_sub03 |
