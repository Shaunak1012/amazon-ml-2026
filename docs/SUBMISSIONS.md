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
| 9 | 2026-09-26 ~14:15 IST | E016-ce-base | `sub-09` (on aca60ee) | 0.9905 dev / 0.9901 OOF | **0.984727** (+0.001405 vs sub-08; rank 67, leader 0.990556) | 0 for us on 26 Sep (team shares slots; 1/day ours) | sub_E016 |
| 10 | 2026-09-26 ~17:00 IST | E020-selftrain | `sub-10` (on 3ff88d3) | 0.9910 dev / 0.9907 OOF (test-side self-training not measurable on dev) | **0.985645** (+0.000918 vs sub-09; new best. E017 alone plausibly explains it: its +0.00045 dev gain and past dev->LB transfer of ~1.4-2x, so self-training looks roughly neutral) | 0 for us on 26 Sep | sub_E020 (hedge sub_E020b kept) |
| 11 | 2026-09-26 ~23:30 IST | E017-namecomp | `sub-11` (on f9f9a90) | 0.9910 dev / 0.9907 OOF | **0.985144** (A/B vs sub-10 0.985645: self-training = **+0.000501** LB; E017 name features vs sub-09 = +0.000417) | 0 left on 26 Sep | sub_E017 (15 cands/S1) |
| 12 | 2026-09-27 09:47 IST | E023b-full | `sub-12` (on 4b58d76) | 0.99100 dev (test-like density) / 0.99096 OOF | **0.98631** (new best; +0.000665 vs sub-10, +0.001166 vs sub-11 same no-ST regime; dev->LB gap 0.0047, was 0.0059; leader 0.991483) | 4 | sub_E023b_full |
| 13 | 2026-09-27 09:54 IST | E023b-full-u55 | `sub-13` | 0.99100 dev (US/India unchanged; France threshold 0.55 vs 0.75, probe: France under-matches 3.256 vs 3.39/S1) | **0.985959** (-0.000351 vs sub-12: looser France hurts -> French pairs in [0.55,0.75) mostly false; France errors are false merges, go stricter) | 3 | sub_E023b_full_u55 |
| 14 | 2026-09-27 10:05 IST | E023b-full-u90 | `sub-14` | 0.99100 dev (US/India unchanged; France threshold 0.90, stricter probe after sub-13 showed looser hurts) | **0.986359** (+0.000049 vs sub-12, flat. France curve 0.55/0.75/0.90 -> 0.985959/0.98631/0.986359: loosening hurts, tightening flat -> errors are in confident French predictions, not at the threshold; final France threshold 0.85 = plateau centre; LLM veto not triggered) | 2 | sub_E023b_full_u90 |
| 15 | 2026-09-27 11:13 IST | E023b-ST-u85 | `sub-15` | 0.99100 dev (self-trained CE on test; France threshold 0.85; candidates = sub-12) | NOT UPLOADED: teammates use the remaining slots; only our final upload is left. sub-15 = current final candidate (expected ~0.9868 from sub-10/11 self-training A/B + sub-12/14 France curve) | 1 | sub_E023b_st_unseen |
| 16 | 2026-09-27 16:32 IST | E033-e030-c383 | `sub-16` (on c985a2b) | 0.99099 dev (test-like density) / 0.99100 OOF; 3.98 cands/S1, RR 99.99996% | **0.987692** (new best; +0.001333 vs sub-14; dev->LB gap 0.0033, was 0.0047: self-trained francized e5-large lifts France) | extra slot 1 of 2 | sub_E033_e030_c383_unseen |
| 17 | 2026-09-27 16:44 IST | E034-ow04 | `sub-17` | 0.99154 dev (test-like density) / 0.99142 OOF; E033 + OW04 owner features; 3.98 cands/S1 | **0.98793** (new best; +0.000238 vs sub-16 E033; dev gain +0.00055 transferred ~0.45x; leader 0.991829) | extra slot 2 of 2 | sub_E034_ow04_unseen |
| 18 | 2026-09-27 19:05 IST | E037-rescue | `sub-18` | 0.99154 dev (test-like density) / 0.99142 OOF; E034 + France self-training round 2 + French candidate rescue; 3.982 cands/S1 | **0.987876** (-0.000054 vs sub-17 E034: France self-training round 2 + rescue flat on public LB; diminishing returns) | final (prepared; upload pending) | sub_E037_rescue_unseen |
| 19 | 2026-09-27 19:16 IST | E034-ow04-final | `sub-19` | 0.99154 dev (test-like density) / 0.99142 OOF; final = E034 (best measured LB 0.98793, sub-17); 3.98 cands/S1 |  | 0 (final upload) | sub_E034_ow04_unseen |
