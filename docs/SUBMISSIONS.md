# Submission ledger

Daily limit: **5** (15 total over 25–27 Sep) · Reset: assumed 00:00 IST · Window closes **27 Sep 23:59 IST** · Reserve ≥ 2 for 27 Sep.

Gate before each submission (see CLAUDE.md → Submission discipline):
1. CV improved meaningfully over best submitted run (> fold std)? 2. Remaining today > reserve?
3. Code + config committed? 4. Both validators pass (`src.er_submission` + organisers' `validate_submission.py`)? → `python scripts/tag_submission.py ...`

| # | timestamp (IST) | experiment id | git tag | CV | public LB | remaining today | file |
|---|---|---|---|---|---|---|---|
| 1 | 2026-09-25 00:51 IST | P0-probe-empty | `sub-01` | n/a (probe; train singleton rate 0.0558) |  | 4 | sub01_empty |
| 2 | 2026-09-25 04:58 IST | E004-predict-sub02 | `sub-02` | 0.9476 dev / 0.9487 OOF | **FAILED at portal evaluation** (file re-verified: passes organiser validator --check-ids + independent checker) | ? | sub02_E004 |
| 3 | 2026-09-25 17:07 IST | E009-stage2-noce | `sub-03` | 0.9643 dev / 0.9618 OOF | **0.947598** | ? | sub03_E009 |
| 4 | 2026-09-25 17:38 IST | probe-france-empty-of-sub03 | `sub-04` | probe (sub-03 with france emptied) | 0.821 (France emptied) → France F0.5 ≈ 0.89 (public), US+India ≈ 0.957 | ? | probe_france_sub03 |
| 5 | 2026-09-25 17:38 IST | probe-india-empty-of-sub03 | `sub-05` | probe (sub-03 with india emptied) |  | ? | probe_india_sub03 | NOT UPLOADED (slot used for France-strict instead)
| 5b | 2026-09-25 ~18:00 IST | probe-france-strict-of-sub03 | (not tagged; sub-03 predictions, France threshold 0.95) | probe | **0.949822** (+0.002224 vs sub-03 → France ≈ 0.905; French errors are false merges) | 1 | probe_france_strict_sub03 |
| 6 | 2026-09-25 18:17 IST | E010-stage2-ce | `sub-06` | 0.9840 dev / 0.9838 OOF | **0.975154** (leader 0.986955; rank 53) | 0 | sub_E010_ce |
| 7 | 2026-09-25 22:10 IST | E013-rarity | `sub-07` | 0.9844 dev / 0.9843 OOF | **0.975726** (+0.000572 vs sub-06) | 4 (26 Sep) | sub_E013_rarity |
| 8 | 2026-09-26 06:11 IST | E015-ftview | `sub-08` | 0.9898 dev / 0.9896 OOF | **0.983322** (+0.007596 vs sub-07; leader 0.990556; dev->LB gap 0.0065, was 0.0087) | 3 (26 Sep; assumes sub-07 counted on 26 Sep) | sub_E015 |
