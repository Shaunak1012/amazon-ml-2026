# Shreyas branch log (teammate tracking + own experiments)

Newest first. Each teammate-sync entry: what they tried, what worked, what failed, and what this branch does
differently because of it.

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
