# Shreyas branch log (teammate tracking + own experiments)

Newest first. Each teammate-sync entry: what they tried, what worked, what failed, and what this branch does
differently because of it.

## 2026-09-27 16:50 IST: team sync + RF01 (train stage 2 on the dev S1s too?) queued
- Shaunak: sub-16 = E033 LB **0.987692**; E034 = E033 + our OW04 owner features: dev **0.99154** (+0.00055 vs E033),
  OOF 0.99142, sub-17 LB **0.98793** (new best, +0.00024). E035 (E034 + CE03) chained, gated vs E034.
- Found while reading his stage 2: test is predicted by the 4 OOF models of the 300k fit S1s (each ~225k); the 100k
  dev S1s never train. RF01 measures whether adding them helps or hurts, candidate rows unchanged
  (`src/er_refit.py`, `er_frames2 --refit-check`, job `scripts/sm/jobs/rf01_refit.sh`): dev split into fixed halves,
  train with the other half added, score this half paired with the base arm; 2 partition seeds (noise floor); OOF
  ensemble ("add") and one full refit at fixed rounds ("full", x1.00/x1.25); calibration, threshold transfer and
  test decision drift per country (France at 0.85) as detriment checks. Two regimes: E034-like (E023b density,
  3.83 filter, OW04) and our final-v2. Results below when in.
- **RF01 result (17:25): no gain, no harm; not worth a slot.** Paired deltas on 50k dev S1 per half, +50k S1 in fit:
  | regime / half | seed noise (g1-g0) | add g0 | add g1 | full x1.00 vs base | full x1.00 vs add |
  |---|---|---|---|---|---|
  | E034-like A | +0.00004 | +0.00001 | +0.00006 | -0.00005 | -0.00006 |
  | E034-like B | +0.00010 | +0.00009 | +0.00007 | +0.00005 | -0.00004 |
  | final-v2 A | -0.00001 | -0.00005 | +0.00002 | -0.00004 | +0.00001 |
  Every 95% CI spans 0; "add" ~+0.00004 on average = seed-to-seed noise. Detriment checks clean: log loss improves
  by 0.00005-0.00017 in all 6 add arms (full refit is slightly worse calibrated than add), OOF-dev gap, matches/S1,
  empty share and the dev-optimal threshold unchanged; early stopping lands at ~100 rounds (lr 0.1) in every arm, so
  stage 2 is data-saturated at 300k S1. Side result: two partition seeds of the SAME model differ by up to 0.00010
  on 50k dev S1, so a +0.0001 dev gate (E035) is at the noise level.

## 2026-09-27 15:00 IST: FINAL v2 (OW04) = v1 within noise
Our final regime, v2 (OW04 e5-large owner) vs v1 (OW03): normal dev 0.99125 vs 0.99117 (+0.00008), test-like dev
0.99098 vs 0.99094 (+0.00004). Both within noise; OW04 ahead in all 4 paired comparisons today (+0.00004..+0.00011,
India-driven). v2 built and organiser-validated (s3://.../shreyas/artifacts/final_shreyas_v2/). Either is defensible.

## 2026-09-27 14:40 IST: paired comparisons in Shaunak's E023b regime
Regime copied (read-only from origin/shaunak, no merge) into src/er_frames2.py: `density_mask` (all fit + dev S1s
kept + random 0.78 of the other train S1s as competitors, seed 11, draws over the chunk-order S1 list; competition
features recomputed on that population, rarity not), and E023b's candidate filter applied AFTER features
(`--prune-post`): stage-1 p >= 0.2 OR E016 CE >= 0.01 (E016 train scores = ce_score_2 of the E019 frame).
Identical dev rows in every arm (100,000 dev S1; 4.706 cand/S1; recall 0.99628 = E023b's reported recall).

| arm | dev F0.5 | vs base | US | India |
|---|---|---|---|---|
| (a) baseline (no norm2, no LLM reranker feature) | 0.99024 | - | 0.98955 | 0.99129 |
| (b) + OW03 owner features (e5-base, GPU) | 0.99107 | +0.00083 | 0.99037 | 0.99213 |
| (c) + CE03 (e5-large cross-encoder, GPU) | 0.99057 | +0.00033 | 0.99004 | 0.99139 |
| (d) + OW03 + CE03 | 0.99126 | +0.00102 | 0.99068 | 0.99215 |
| (e) + OW04 owner features (e5-large, GPU) | 0.99115 | +0.00091 | 0.99040 | 0.99230 |
| (f) + OW04 + CE03 | **0.99137** | **+0.00113** | 0.99060 | 0.99253 |

- Baseline sits below E023b-full (0.99100) because our frames lack his norm2 features and E022 reranker score
  (together ~+0.00076 in his runs: E025 0.99039 -> E023b 0.99100 with the filter).
- The owner signal is worth ~3x more at test-like density (+0.00083) than at train density (+0.00026).
- If roughly additive: E023b + OW03 + CE03 ~ 0.9920 dev (medium confidence). Features ready for him as parquet keyed by
  (s1_id, cand_id): OW03 train/test_owner, CE03 train/test ce3 (fold-0 + test, trained on folds 1-4 only).
- OW04 vs OW03 (same rows): +0.00008 alone, +0.00011 with CE03; India +0.00038, US -0.00008 in (f) vs (d).
  Consistent direction but at the edge of noise. Best arm: (f) OW04 + CE03 = 0.99137 (+0.00113 over base).

## 2026-09-27 13:05 IST: FINAL v1 built (clean, 4.54 cand/S1, organiser validator PASS)
- CE03 (e5-large CE, GB10): alone +0.00006 dev; with OW03 0.99134. As a candidate filter (p >= 0.05 OR ce3 >= 0.10):
  4.13 cand/S1 on dev at 0.99131 (-0.00003), recall 0.9908.
- Self-trained ce_score_2 (E020) dropped from features (team decision): 0.99131 -> 0.99117 normal dev.
- Test-density training (19% S1 drop): +0.00034 on test-like dev (FB 0.99094 vs FA_e 0.99060) -> kept.
- **FINAL v1** = E019 train frame + E020 test frame minus ce_score_2, + OW03 + CE03 features, cascade filter applied to
  train and test with population features recomputed, density training, one stage-2 run -> both TSVs.
  Test: 4.54 cand/S1 (US 4.33, India 4.37, France 5.56), 3.37 matches/S1, 5.8% empty. Organiser validator PASS.
  s3://.../shreyas/artifacts/final_shreyas/. Seed-averaging: no gain. OW04 (e5-large owner model) running for a v2.

## 2026-09-27 01:00 IST: OW03 (GPU owner model, e5-base) = +0.00026 dev -> 0.99122; clean filter sweep
- Team decision (26 Sep 22:40): drop self-training; filter before features. E020's ce_score_2 is self-trained on test,
  so the clean candidate rule uses stage-1 p OR the E008 e5-small CE (ce_score): p>=0.02|ce>=0.02 -> 6.25 cand/S1,
  dev 0.99096 (= top-15 0.99097); p>=0.05|ce>=0.05 -> 5.43, 0.99082; p>=0.1|ce>=0.1 -> 4.70 dev / 4.89 test.
- OW03 (GB10, e5-base, 2 epochs, 137k groups; trained in 17 min): clean-filter dev 0.99122 vs 0.99096 (+0.00026,
  US +0.00030, India +0.00020); top-15 0.99122 vs 0.99097. First real gain from the owner signal (OW01 e5-small ~0).
- GPU box: lab GB10 (97.5 TFLOPS bf16), runner in tmux (the box kills SSH-session processes on logout).
- Next: 3-seed stage-2 average; CE03 (e5-large CE) training/scoring; CE03-based filter toward 4.7/S1.

## 2026-09-27 00:05 IST: OW01 owner model (CPU, e5-small, 48k easy groups) = no gain; GPU plan
- OW01 paired dev: top-15 0.99094 vs 0.99097 (-0.00003); strict pruned 0.99100 vs 0.99089 (+0.00010). Within noise.
  First attempt crashed (extra-feature glob matched train_groups.parquet); fixed (explicit <split>_owner.parquet).
- GPU: user's lab GB10 box via a credential-free runner (presigned queue GET + scoped POST; no AWS keys on the box).
  Queued GPU job: OW03 (owner model, e5-base, 2 epochs, 137k hard groups) then CE03 (multilingual-e5-large
  cross-encoder, MIT, 560M; trained on fold 1-4 hard pairs, scores the strict candidate set: 2.03M fold-0 + 8.99M
  test pairs). r7i compares each automatically. Final run now prunes TEST with the same rule and recomputes population
  features on the pruned test set (candidate_pairs.tsv == what stage 2 scores).
- LLM reranker not used: 7B prefill over millions of pairs is too slow on one GPU, and Shaunak runs E022.

## 2026-09-26 23:00 IST: CS03 France check, SH01 submission built, JD01 dropped, owner model (OW01/OW02) running
- **CS03 (real test, E020 frame): the stage-1 floor does NOT hurt France.** Test cand/S1 at floor 0.005 / 0.01 /
  0.02: India 6.30 / 5.60 / 4.94, US 6.23 / 5.55 / 4.91, **France 7.91 / 6.81 / 5.83**. French stage-1 probs are
  less peaked, so the floor removes fewer French candidates, not more.
- **SH01 submission** (E020 test frame + 4-model test-density stage 2, expected-F): both validators PASS;
  94.3% non-empty, 3.37 matches/S1 (US 3.38, India 3.38, France 3.32). s3://.../shreyas/artifacts/sub_SH01/.
  Top-15 candidates (not pruned). Expected LB gain over E020 small (+0.0003-0.0004 from dev).
- **JD01 (joint/release decoding) dropped:** on E016 test probs, records rejected by their top S1 whose 2nd S1 has
  p >= 0.5: 5,913 (0.33% of S1); p >= 0.7: 64; p >= 0.9: 0. No confident second owner exists to release to.
- **GPU:** g5.2xlarge training-job quota is 0 (spot and on-demand); Studio g5 capacity unavailable. CPU only.
- **OW01/OW02 listwise owner model** (`src/er_owner.py`): record + up to 6 competing S1s in one sequence
  (multilingual-e5-small, CPU, fp32 train / bf16 infer), softmax over slots + none -> stage-2 features own_p,
  own_margin, own_none, own_n (`er_frames2 --extra-feats`). OW01 leakage rule was too strict (48k groups); OW02 uses
  the slot-list rule (137k of 300k contested groups, 3.13 slots/record). OW01 slot accuracy 0.85 at step 250.

## 2026-09-26 20:05 IST: CS04 strict OR-rule candidate sets -> recommendation
Shaunak's pruning (a6df68d) filters stage-2 rows AFTER building features on the full top-15, so competition/sibling
features and the OR-rule CE use pairs that are not in candidate_pairs.tsv (reviewers read the code: risk that the
"real" candidate set is judged to be 15). CS04 measures the OR rule in strict form (every stage-2 population feature
recomputed on the pruned set; CE = E016 scores, ce_score_2 in the E019 frame):

| rule (strict) | cand/S1 | recall | dev F0.5 | vs top-15 |
|---|---|---|---|---|
| p >= 0.01 OR CE >= 0.01 | 5.69 | 0.9968 | 0.99092 | -0.00005 |
| p >= 0.02 OR CE >= 0.01 (Shaunak's) | 5.28 | 0.9966 | 0.99088 | -0.00009 |
| **p >= 0.02 OR CE >= 0.02** | **5.08** | **0.9961** | **0.99089** | **-0.00008** |
| p >= 0.05 OR CE >= 0.01 | 4.95 | 0.9964 | 0.99078 | -0.00019 |

- The CE OR-rule dominates stage-1-only floors (same size, recall 0.996 vs 0.994, a third of the loss).
- Strict costs ~nothing vs his non-strict "unchanged" (within noise), so use strict: no reviewer ambiguity.
- **Recommendation: p >= 0.02 OR CE >= 0.02, features on the pruned set: 5.08 cand/S1 (3x smaller) at -0.00008.**
  France pending CS03 (needs the E020 test frame). Caveat: train-side competitors outside fold 0 have no CE score and
  fall back to the prob rule.

## 2026-09-26 19:40 IST: CS02 low floors
| floor | cand/S1 | recall | dev F0.5 | vs top-15 |
|---|---|---|---|---|
| 0.003 | 6.43 | 0.9961 | 0.99087 | -0.00010 |
| 0.005 | 5.93 | 0.9954 | 0.99083 | -0.00014 |
| 0.0075 | 5.53 | 0.9944 | 0.99079 | -0.00018 |
| 0.01 | 5.25 | 0.9936 | 0.99073 | -0.00024 |
| top-10 + 0.005 | 5.77 | 0.9949 | 0.99081 | -0.00016 |

0.003-0.01 are within noise of each other; knee at ~0.01. **Recommendation: floor 0.005 (5.9/S1, 2.5x smaller)**, or
0.01 (5.25/S1) if candidate size is weighted more. France is unmeasured (stage 1 never saw French data): CS03 checks
test candidates/S1 by country and the share of E020-accepted pairs each floor would cut, once the E020 frame lands.

## 2026-09-26 19:20 IST: CS01 candidate-set size vs F0.5 (organisers' update: smaller candidate sets rank higher)
E019 train frame, normal density, competition features recomputed on the pruned population, stage 2 refit per
setting. Dev = all 100k dev S1 (S1s left with no candidates are scored as predicted empty).

| setting | cand/S1 | recall | dev F0.5 | vs top-15 |
|---|---|---|---|---|
| top-15 (current) | 15.00 | 0.9974 | 0.99097 | - |
| top-10 | 10.00 | 0.9963 | 0.99087 | -0.0001 |
| top-8 | 8.00 | 0.9940 | 0.99076 | -0.0002 |
| top-6 | 6.00 | 0.9767 | 0.98896 | -0.0020 |
| top-5 | 5.00 | 0.9425 | 0.98443 | -0.0065 |
| **stage-1 p >= 0.01** | **5.25** | 0.9936 | **0.99073** | **-0.00024** |
| top-8 + p >= 0.01 | 5.06 | 0.9919 | 0.99063 | -0.0003 |
| p >= 0.02 | 4.62 | 0.9905 | 0.99035 | -0.0006 |
| p >= 0.05 | 4.06 | 0.9854 | 0.98985 | -0.0011 |

- A stage-1 probability floor (adaptive per S1) dominates fixed caps: p >= 0.01 cuts candidates 65% (15 -> 5.25/S1)
  for -0.00024 dev F0.5; fixed caps below 8 break entities with many true matches. CS02 (p 0.003-0.0075) running.
- Final pipeline: apply the floor to the stage-1 output before CE scoring and stage 2, so candidate_pairs.tsv equals
  the set stage 2 scores. Doc: present stage 1 as a learned blocking filter (cascade allowed by the rules).

## 2026-09-26 18:40 IST: sync - Shaunak's E025 = SH01 in the main pipeline
**shaunak @ adafc7d:** sub-10 (E020, self-trained CEs) **LB 0.985645** (new best). E024 `--norm2` (street types
expanded, dotted legal forms). **E025 `--comp-keep 0.78`**: thins never-fitted train S1s so competition features see
test density (his count: 2.67 vs 3.27 S1s per record, i.e. keep ~0.82; ours from pool sizes: keep 0.81 = two
independent routes, same number). His version keeps all fit/dev S1s (better than SH01, which lost 19% of fit S1s).
- SH01 as a submission is superseded by E025 (same idea + norm2 + E020 CE). Kept as fallback job only.
- Notes for E025: (1) its dev frame is also thinned, so its dev F0.5 is NOT comparable to E024's (density alone
  moves dev ~-0.001); compare against the current recipe scored in the same thinned dev (SH01 `eval` arm), where we
  measured +0.0002..0.0004. (2) Averaging 3-4 thinning seeds added ~+0.0001. (3) Name-rarity counts should also use
  the thinned S1 population (er_frames2.repopulate does).
- Next on this branch (no overlap): stage-2 GBDT bake-off (CatBoost / XGBoost / LightGBM seeds) on the cached frames.

## 2026-09-26 18:20 IST: SH01 robust + ensemble
Paired gain (SH01 both minus current recipe, same dev S1s) on 3 independent test-like dev subsets:
seed 11: expected-F +0.00029, thr .75 +0.00020 · seed 12: +0.00045 / +0.00030 · seed 13: +0.00033 / +0.00018.
**Positive in 15/15 (subset x rule) comparisons.** Averaging 4 test-like models (drop seeds 11/12/13 at 19% + one at
30%, same dev): expected-F 0.98999 vs 0.98957 current (**+0.00042**). A 30% drop alone is worse than 19%, which
matches test density. Absolute dev moves ~0.0003 between subsets: compare only paired.
**Final-candidate stage 2 = this 4-model average** on the E020 test frame (pending Shaunak's upload).

## 2026-09-26 17:40 IST: SH01 result = density shift confirmed, test-like training gives a small real gain
Frames: Shaunak's E019 train frame (fit = 300k fold-0 S1, dev = 100k). Test-like dev = dev minus a random 19% of all
train S1s (their records become distractors), competition + rarity features recomputed. Same 80,956 dev S1 for every arm.

| arm | expected-F | thr .65 | thr .70 | thr .75 | thr .80 |
|---|---|---|---|---|---|
| current recipe on NORMAL dev (E019) | 0.99081 | | | | |
| eval: current recipe | 0.98957 | 0.98924 | 0.98945 | 0.98972 | 0.98981 |
| evalsub: current recipe, fit on same 243k S1 as both | 0.98949 | 0.98920 | 0.98941 | 0.98962 | 0.98974 |
| **both: stage 2 fit in the test-like population** | **0.98986** | **0.98968** | **0.98987** | **0.98992** | **0.98991** |

- Density alone costs the current model **-0.0011** (0.99081 -> 0.98972): a real part of the US/India dev->LB gap.
- Test-like training wins under every rule, in both countries (US +0.0001, India +0.0002 at the chosen rules) and
  against the fit-size-matched control: **+0.0002 to +0.0004**. It mostly acts as a learned stricter operating point
  (current model's best threshold drifts 0.75 -> 0.80 under density), learned from train data rather than guessed.
- Decision: keep for the final candidate (same features/pipeline, low risk). Running: drop-seed ensemble (fixed dev)
  and 30% drop. Remaining density loss needs new features, not retraining.

## 2026-09-26 17:10 IST: SH02 decision layer = no gain (dropped)
Exact expected-F0.5 (Poisson-binomial, `src/er_decide2.py`) vs the current 256-sample Monte-Carlo, on Shaunak's E019 dev
predictions (100k S1): MC256 **0.99081**, exact **0.99082**, exact + 2-fold isotonic **0.99079**. All within noise:
stage-2 probabilities are already calibrated and 256 samples suffice. The decision layer is not where the loss is;
the remaining levers are upstream (stage-2 training regime, SH01). Global assignment decoding stays parked: dev files
hold only dev S1s, so conflicts with other S1s cannot be measured.

## 2026-09-26 14:45 IST: SH01 distractor simulation (new lever) + sync
**Teammate sync (shaunak @ ea738d6):** E016 (both CEs) dev 0.9905 → **LB 0.984727 (sub-09)**, rank 67; leader still
0.990556. Dev→LB gap is still ~0.006. Shaunak now runs **E018 = French CE self-training on test pseudo-labels** and
E019 = best features + E018 CE + cached frames (`--frames runs/frames/E019`). → This branch **drops its France
self-training plan** (duplicate) and takes the gap that nobody is working on:

**Finding: test has ~2× more unmatched pool records per S1 than train.** The generator is very uniform: train has
3.46 matches/S1 and 5.6% singletons in both countries, and 26% of pool records are unmatched. Pool/S1 is 4.68 in train
but 5.76 (US), 5.82 (India) and 5.53 (France) in test. With the same 3.46 matches/S1 (team test predictions ~3.2/S1 are
consistent with this), test has **~2.3 distractors per S1 vs 1.2 in train**. Stage 2's record-level features (best
OTHER S1, margin) and the decision threshold were learned in the sparser world, so test produces more false merges
on distractors whose own S1 is absent. That fits a US/India dev→LB gap that shrinks with better matchers but never
closes.

**SH01 (code ready, tests green):** drop 19% of train S1s (1 - 4.68/5.75) so their matched records become
distractors, recompute competition + rarity features over the reduced population, refit stage 2. Two arms on the
same test-like dev: `--drop-in eval` (current recipe, fair baseline) vs `--drop-in both` (test-like training).
Runs on the SageMaker CPU box from E019's cached frames (`src/er_frames2.py`), so it does not touch the GPU box queue;
also available in `er_fullpass stage2 --drop-s1-frac/--drop-in`.

## 2026-09-26 14:30 IST: sync + plan
**Team state (main/shaunak @ aca60ee/f9f9a90):** best LB sub-08 **0.983322** (E015: fine-tuned bi-encoder view +
stage 1/2 on fold 0; dev 0.9898). E016 (second CE, e5-base) and E017 (name competition features) are running on the
GPU box. Leader 0.9906.
- Worked: full-population competition features (+0.017), cross-encoder feature (+0.020), fine-tuned bi-encoder
  retrieval view (+0.005), name rarity (+0.0004).
- Didn't work: two fold-disjoint CEs (E011), stage 3 (E015), char-ngram retrieval view alone (small).

**Where the LB gap is (estimate, medium confidence):** test S1 = India 46.7% / US 38.3% / **France 15.0%** (train has
no France). Dev by country (E015) US 0.9892, India 0.9907, so the mix-weighted dev is 0.9900. With a US/India
dev→LB gap of ~0.003 (as measured at sub-03), sub-08's 0.9833 implies **France ≈ 0.96**. France alone then costs about
0.0045 LB, and it is the lever nobody on the team is working on.

**France observations (test EDA):** names reuse a small vocabulary ("Fédération", "Boxing", "Vercors") → the address
decides. French noise the model never saw in training: street-type abbreviations (R/RUE, AV/AVE, BD/BLVD, CH, ALL, CITÉ),
"N°/NO." prefixes, zero-padded numbers (0013), département ↔ région swaps (Nord ↔ Hauts-de-France, Gironde ↔
Nouvelle-Aquitaine, Loire-Atlantique ↔ Pays de la Loire), legal forms EI / Ets / "(France)" missing from
`LEGAL_TOKENS`. `er_normalize.normalize_address` expands no abbreviations for any country.

**This branch's plan:** (1) France adaptation on CPU from the GPU box's exported stage-1/CE features: label-free
French address canonicalisation learned from the test pool itself, plus pseudo-label self-training of a
France-specific stage 2; one clean LB A/B (France rows only changed). (2) Decision-layer work validated on dev
predictions (applies to every country).
