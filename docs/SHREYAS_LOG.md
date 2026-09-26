# Shreyas branch log (teammate tracking + own experiments)

Newest first. Each teammate-sync entry: what they tried, what worked, what failed, and what this branch does
differently because of it.

## 2026-09-26 15:30 IST: SH01 distractor simulation (new lever) + sync
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
