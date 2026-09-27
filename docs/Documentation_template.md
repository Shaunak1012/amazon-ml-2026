# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** SHSHSHSH  
**Team Members:** Shaunak A. Rai, Shaswat Solanki, Shivam Anand, Shreyas Sreenivas  
**Submission Date:** 27 September 2026

*Sections 1-6 are the core write-up (about two pages). The appendices hold the detail: experiment log, features,
hyperparameters and compute, error analysis, models and licences, submission history, and the code map. Numbers are
labelled **dev** (100k held-out train S1s that no model saw), **OOF** (out-of-fold on the stage-2 fit pool) or
**LB** (public leaderboard).*

---

## 1. Executive Summary
We use a cascade that starts with fast, high-recall steps and ends with precise, expensive ones. First, same-country
dense retrieval over four multilingual-e5 views, one of them a bi-encoder we fine-tuned on train matches. Then a
LightGBM filter, three fine-tuned cross-encoders (multilingual-e5 small, base and large), and a Qwen3-4B LoRA reranker that sees the *competing* S1s. Last, a
stage-2 LightGBM and an F0.5-tuned decision layer that gives each S2/S3 record to at most one S1. The final model runs
on **3.83 candidates per S1 on dev (3.98 on test) at 99.1% pair recall** and scores **0.99154 dev / 0.98793 public
LB**. The main idea is that features must mean the same thing on train, dev and test:
cross-S1 features are computed over the full S1 population at test-like competitor density, and every learned score
is out-of-sample on the S1s used to fit and validate the final model. For France, the country absent from train,
we adapt the cross-encoders with test-time self-training on our own confident predictions (allowed by the organisers)
and with synthetic French-style training pairs built from the provided train records.


![Final pipeline (E034)](architecture.png)

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** train has 2.21M S1, 5.03M S2 and 5.29M S3 records, with 7.64M matched pairs. Test has 1.73M S1 and
  9.97M S2+S3 records. All-pairs comparison is impossible.
- **Structure (train):** 5.6% of S1s are singletons. There are 3.46 matches per S1 on average, 0 records matched to
  more than one S1, 0 cross-country matches, and ~26% of S2/S3 records match no S1. We turned these facts into two
  design rules: a hard one-S1-per-record constraint, and blocking within the country string.
- **Noise:** ~24% of Indian S2/S3 names are in Indic scripts (Devanagari, Kannada, Telugu, Gujarati). Some matched
  names are replaced by unrelated strings while the address survives, so **the address is a first-class signal**.
  Addresses can be empty or contain literal `null`. We also see leetspeak, moved legal suffixes, "F/K/A" names and
  reordered address parts. 34% of test S1 names occur verbatim as train S1 names, so a name alone does not identify
  a business.
- **Shift:** France (15% of test S1s) never appears in train. Test has 5.75 S2+S3 records per S1 against 4.68 in
  train. In test each record is retrieved by 2.67 (US) to 2.84 (France) S1s, against 3.27 in train. This matters for
  every feature that compares a record's S1s.

### 2.2 Solution Strategy
**Approach Type:** Hybrid: blocking (dense retrieval), a cascade of classifiers (LightGBM, fine-tuned transformer
cross-encoders, an LLM reranker) and an explicit F0.5 decision layer.  
**Core Innovation:** (1) A **fine-tuned bi-encoder retrieval view** that recovered 90.7% of the true pairs the
off-the-shelf views missed. (2) **Full-population, density-matched competition features.** Every train and test S1 is
scored, so "how well does this record match its *other* S1s" is computed the same way everywhere. Train is thinned
to test-like density. (3) An **LLM reranker whose prompt lists the competing S1s**, which targets the dominant
remaining error: look-alike names with empty addresses.

### 2.3 Validation Design
- 5 random folds over train S1s (seed 42). The **bi-encoder, cross-encoder B and the LLM train on folds 1-4**, and
  cross-encoder A on folds 1-2. None of them sees fold 0.
- **Stage 1 and stage 2 are fitted on fold 0 minus dev** (341k S1, 4 internal OOF groups). Dev is a fixed 100k-S1
  sample of fold 0, used only for scoring. So every neural score is out-of-sample on the fit pool and on dev, exactly
  as on test.
- Retrieval uses the **full** train S2/S3 pool (same size as test). Stage 1 scores **all** train S1s (out-of-fold) and
  all test S1s, so cross-S1 features see the full population. An early check that scored only sampled S1s (E005)
  gave a biased estimate: 31% of records were contested there, against 87% on test.
- **Test-like density:** from E025 on, the population used for competition features is thinned (78% of never-fitted
  S1s kept). We report dev at that density, the fairest dev to compare with the LB.
- France has no dev signal. Public-LB probes measured it (Appendix B.4). We tracked the dev→LB gap on every upload:
  0.0087 → 0.0065 → 0.0054.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** equality of the `country` string (a partition, no country list is hard-coded), then exact
  GPU top-10 cosine search per view and per source (S2, S3). There are four views:
  multilingual-e5-small on *name*, *address* and *name + address* (raw text), and the same model **fine-tuned as a
  bi-encoder** (in-batch InfoNCE, same-country batches, 1.5M train pairs of folds 1-4) on name + address.
  The union of all hits is the retrieved set. A stage-1 LightGBM on 32 cheap features (embedding cosines, rapidfuzz
  name/address similarities, number/postcode agreement, rank context) then filters it.
- **Candidate pairs generated:** 3.83 per S1 on dev (pair recall 0.991); 3.98 per S1 (US 3.76, India 3.86, France
  4.92; at most 15) and 6,895,815 pairs on test. The reduction ratio against all S1 × (S2 ∪ S3) pairs is
  99.99996%. Final filter: stage-1 probability ≥ 0.5 OR cross-encoder (E016) score ≥ 0.05.
  Tightening it from 4.71 to 3.83 per S1 cost only 0.00005 dev F0.5, because the dropped pairs were almost never
  matched. Stage 2 is fitted and applied on exactly this set, so `candidate_pairs.tsv` is the matcher's inference set.
- **Candidate generation efficiency (test, 1.73M S1 entities, 9.97M S2/S3 records):**

| Stage | Pairs per S1 | Pairs in total | Work | Pair recall (dev) |
|---|---:|---:|---|---:|
| All S1 × (S2 ∪ S3) pairs | 9.97M | 1.7 × 10^13 | not computed | 1.0 |
| A. same-country dense k-NN (4 views × 2 sources × top-10) | ~62 | ~1.1 × 10^8 | one embedding per record (linear) + k-NN per S1 (ANN at scale) | 0.9976 |
| B. stage-1 LightGBM on cheap features, top 15 | 15 | 2.6 × 10^7 | ~3 ms per S1 incl. features (CPU) | 0.9974 |
| C. filter: stage-1 ≥ 0.5 OR e5-base CE ≥ 0.05 | **3.98** | **6.9 × 10^6** | CE on the 15 kept pairs (GPU) | **0.991** |

  The matcher (and every expensive model after the filter) runs on 3.98 pairs per entity; the whole cascade is linear
  in the number of S1 entities. Measured on one RTX 5080: e5-large CE scoring 1,460 pairs/s, stage 2 ~30 min CPU.
- **Scalability (billions of records):** no step compares all pairs. Retrieval is dense-vector k-nearest-neighbour
  search per view, partitioned by country; at competition size we ran it as an exact chunked GPU inner product to
  maximise recall, and at production size the same embeddings go into an approximate index (FAISS IVF-PQ or HNSW,
  sharded by country/region), which answers top-k in roughly O(log M) per query with the same k. Every later stage
  (stage-1 LightGBM, cross-encoders, reranker, stage 2) runs only on the ≤ 15 retrieved pairs per S1 and then on the
  final ~3.8, so the cost is linear in the number of S1 records.

| Stage (dev, 100k S1) | Rule | Candidates / S1 | Pair recall |
|---|---|---:|---:|
| 3 off-the-shelf views (E003-E013) | top-10 per view per source | ~50 | 0.9740 |
| A. Dense retrieval, 4 views (E015) | + fine-tuned view | ~62 | 0.9976 |
| B. Stage-1 LightGBM | top-15 by stage-1 prob | 15 | 0.9974 |
| C0. Earlier final set (sub-12 to sub-15) | stage-1 prob ≥ 0.2 **or** cross-encoder B ≥ 0.01 | 4.71 | 0.9963 |
| **C. Final set = `candidate_pairs.tsv`** (the set fed to the matching model) | stage-1 prob ≥ 0.5 **or** cross-encoder B ≥ 0.05 | **3.83** | **0.9910** |

A and B are internal blocking steps; `candidate_pairs.tsv` is C, exactly the rows the matching model (stage 2) is
fitted and applied on. Smaller filters cost F0.5 quickly (3.67/S1: -0.00005 more; 3.55/S1: -0.00024) because the
floor is our own 3.37 predicted matches per S1 (every matched ID must be a candidate), so 3.83 is the knee of the curve.

- **How we ensured true matches were not lost:** we measured recall on dev at every stage before choosing a cut-off.
  Error analysis of E010 showed that 51% of the remaining loss was pairs never retrieved (India 3x worse than US), so
  we trained the bi-encoder view (+0.0054 dev F0.5). Stage C is an OR rule, so a pair the cheap model doubts survives
  when the cross-encoder likes it. The final set still has a **theoretical ceiling of 0.99892 dev F0.5** (oracle
  decisions on these candidates), far above our score, so blocking is no longer the bottleneck. The stage-2 model is
  fitted and applied on exactly the stage-C rows. Its context features (siblings, competitors) are computed from the
  stage-B top-15 lists. At billion scale the exact search would become an ANN index (e.g. HNSW) per country
  partition; nothing else changes.

---

## 4. Matching Model

**Features used** (80 in stage 2; full list in Appendix B.2):
- **Name features:** rapidfuzz ratio, token-set, token-sort, partial and Jaro-Winkler on normalised and "core"
  (legal-suffix-stripped) names; exact core-name match; lengths. The same features on v2 strings (dotted legal
  forms collapsed). Name-view embedding cosine. Label-free name rarity (how many S1s and records share the name,
  rarest-token frequency).
- **Address features:** token-set/sort/partial similarity, empty-address flag, and v2 similarities after street-type
  expansion (St/Street, R/Rue). House-number Jaccard, common count and first-number equality. Postcode agreement.
  Address-view cosine.
- **Other:** name + address cosines (off-the-shelf and fine-tuned); retrieval ranks; source flag; rank and gap to the
  best within the S1's and within the record's lists; stage-1 probability. **Three cross-encoder scores** and the
  **LLM reranker P(Yes)** (only in the uncertain band, missing elsewhere). **Cluster features:** similarity to the
  S1's confident candidates, and probability aggregates. **Competition features:** the candidate record's best value
  with any *other* S1 and our margin over it, for 6 name similarities and the stage-1 probability. They are computed
  over the full population at test-like density. **Owner features** (E034): the listwise owner model's probability that
  this S1 owns the record, and its margin over the best competing S1 (only for contested records, missing elsewhere).

**Model type:** a cascade. (1) Stage-1 LightGBM. (2) Cross-encoder A: multilingual-e5-small as a sequence-pair
classifier, 2M pairs, 40% positive, hard negatives chosen by stage-1 probability. (3) Cross-encoder B:
multilingual-e5-base, 3M pairs. (4) **Qwen3-4B with LoRA** (r 16) trained as a Yes/No classifier on 40k prompts. Each
prompt shows the S1, the candidate record and up to 3 other S1s that also retrieved the record. It scores only pairs
with cross-encoder B in (0.1, 0.9): 155k fold-0 and 727k test pairs. (5) Cross-encoder C: multilingual-e5-large, adapted
to the unseen country with francized train pairs and self-training on confident test predictions (E027, E029, E030).
(6) **Listwise owner model (OW04)**, multilingual-e5-large: for a pool record retrieved by several S1s it reads the
record and up to 6 competing S1s in one sequence and predicts which one owns it (softmax over slots + "none"). Pair
models never see the competitors, and missed or wrong owners in name collisions are our dominant error. It is trained
only on groups with no fold-0 S1 involved, so every fold-0 row is scored out-of-sample (+0.00055 dev, +0.00024 public
LB). (7) Stage-2 LightGBM (127 leaves, lr 0.1, early stopping) on all features.  
**Threshold selection method:** on stage-2 OOF predictions only, we compare (a) one global threshold, grid-tuned
for macro F0.5, and (b) a per-S1 expected-F0.5-optimal subset. Both enforce **one S1 per record** (each S2/S3 record
goes to its highest-probability S1). Recent runs chose (a) at 0.75 (global threshold 0.75 with one-S1-per-record assignment). An empty
list is predicted whenever nothing passes, which earns the singleton credit. The unseen-country variant uses
threshold 0.85 for S1s whose country string never occurs in train, from the same run
(the final upload is the unseen-country variant, threshold 0.85; public-LB probes on the same probabilities gave 0.985959 / 0.98631 / 0.986359 at 0.55 / 0.75 / 0.90, so loosening hurts and tightening is flat, and 0.85 is the plateau centre).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.99154 dev at test-like density (OOF 0.99142) and **0.98793 public LB** for the final
  model (E034, uploaded as sub-17 and again as the final). Dev and OOF agree to
  0.00001 in every recent run.

| Step (experiment) | Dev F0.5 | Public LB |
|---|---:|---:|
| Baseline: 3-view retrieval + LightGBM + threshold (E003) | 0.9476 | failed at portal (sub-02) |
| Full-population stage 2 with competition features (E009) | 0.9643 | 0.947598 |
| + cross-encoder A (E010) | 0.9840 | 0.975154 |
| + name-rarity features (E013) | 0.9844 | 0.975726 |
| + fine-tuned bi-encoder view, clean fold-0 fit pool (E015) | 0.9898 | 0.983322 |
| + cross-encoder B, e5-base (E016) | 0.9905 | 0.984727 |
| + name competition features (E017) | 0.9910 | 0.985645 (sub-10, with self-training) |
| + LLM reranker (E022) | 0.99116 | not submitted |
| + test-like density (E025, dev at test-like density) | 0.98975 → 0.99039 | not submitted |
| + v2 normalisation, reranker on the wider band, 4.71/S1 filter (E023b) | 0.99100 | 0.98631 (sub-12) |
| + France threshold 0.90 on the same probabilities | 0.99100 | 0.986359 (sub-14) |
| + e5-large cross-encoder (E027) | 0.99104 | not submitted |
| + francized e5-large (E029) + self-training on it (E030) | 0.99104 | not submitted |
| E033: E030 + 3.83/S1 candidate filter + France threshold 0.85 | 0.99099 | 0.987692 (sub-16) |
| **Final (E034)**: + listwise owner model OW04 | **0.99154** | **0.98793** (sub-17) |
| E037: E034 + France self-training round 2 (E036) + French candidate rescue | 0.99154 | 0.987876 (flat; not final) |

- **Common false positives (wrong merges):** different businesses with the same or near-identical name (reused names,
  chain branches) when the address is empty or generic. **French records** before normalisation v2: "2 R PAUL
  VERLANE" vs "2 RUE PAUL VERLAINE" looked different to every address feature, so names alone decided. The public-LB
  probes showed that French errors are mostly false merges.
- **Common false negatives (missed matches):** records whose name was replaced by an unrelated string and whose
  address is truncated, and Indic-script names with sparse addresses. After the bi-encoder, ~85% of the remaining dev
  loss is **retrieved but rejected** true matches, mostly empty-address records whose name nearly matches several S1s.
  The competition features and the LLM reranker target exactly this.
- **Dev→LB gap** (0.0054 at sub-10, 0.0047 at sub-12 after test-like density). Our estimates of its parts: test-like density ~0.0011 (now modelled), France
  ≥ 0.0024, denser test pool ~0.0002, and model overconfidence on test ~0.001-0.002.

---

## 6. Conclusion
Recall first, then precision. A fine-tuned retrieval view removed most unretrievable pairs. Cross-encoders and a
competitor-aware LLM reranker then attacked the look-alike merges that F0.5 punishes most. We cut the final set to
3.83 candidates per S1 (3.98 on test), 19% smaller than our first final, for 0.00005 dev F0.5. The biggest lessons were about validation, not
models: features that compare a record's S1s must be computed over the full population and at test density, and each
learned score must be out-of-sample where the next stage is fitted. Test-time self-training (confirmed allowed by the
organisers) was the strongest lever for France: the self-trained e5-large cross-encoder cut French uncertain pairs
per S1 from 0.447 to 0.331. Dev is bounded near 0.991 by empty-address records whose name is shared by several S1s
(oracle +0.0053): nothing in such a record identifies its owner, and we did not use ID or row-order signals.


**Production path.** At Amazon scale we would keep the cascade shape (sharded ANN retrieval, a cheap filter to ~4 candidates per entity, one matcher) but distil the three cross-encoders, the LLM reranker and the owner model into one small cross-encoder, compute competition features per blocking shard, calibrate, and route the ambiguous band (~0.5% of pairs: empty-address records shared by several same-name entities) to human review. A new market gets synthetic pairs, a small labelled sample and self-training, a manual audit, and a drift guard on every model update (docs/FRANCE_QA.md).
---

## Appendix

### A. Code Artefacts
The runnable pipeline is in `code/business_entity_resolution/`: all source in `src/`, `README.md` (= reproduction
guide, with runtimes), and `requirements.txt` (pinned; torch 2.8.0 + CUDA 12.8). **Entry point:**
`bash scripts/reproduce.sh` regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the
organisers' TSVs (~24 h, ~20 GPU-hours on one RTX 5080 16 GB, 64 GB RAM) and runs both validators.

| Module | Role |
|---|---|
| `src/er_data.py` | exact TSV loading (tab, no quoting, strings) and parquet cache |
| `src/er_normalize.py`, `src/er_norm2.py` | normalisation v1 (anyascii transliteration, core name, numbers, postcode) and v2 (street types, dotted legal forms) |
| `scripts/make_folds.py`, `src/cv.py` | shared S1 folds (seed 42) and the 100k dev subset |
| `src/er_embed.py`, `src/er_biencoder.py` | e5 embeddings per view; bi-encoder fine-tuning (E014) |
| `src/er_blocking.py`, `src/er_pipeline.py` | same-country exact top-k retrieval, featurisation, decision-rule choice |
| `src/er_features.py` | 32 stage-1 pair features |
| `src/er_fullpass.py` | stage 1 over all S1s; stage 2 (features, candidate filter, LightGBM, decisions, TSVs) |
| `src/er_stage2.py` | cluster and full-population competition features |
| `src/er_crossenc.py` | cross-encoder training and scoring |
| `src/er_llmrank.py` | reranker prompts, LoRA training, scoring, mapping to a feature |
| `src/er_model.py`, `src/er_decide.py` | grouped-OOF LightGBM; threshold / expected-F0.5 decisions with one-S1-per-record assignment |
| `src/metrics.py`, `src/er_submission.py` | official macro F0.5 (unit-tested); writer and validator of both TSVs |

### B. Additional Results

#### B.1 Experiment log (dev = 100k held-out S1; full log in `docs/EXPERIMENTS.md`)
| ID | Change | Result | Kept? |
|---|---|---|---|
| E002 | recall of single views (dev, top-10) | name 0.592, address 0.868, union 0.946 | address view is essential |
| E003 | baseline: 3 views, k=10, LightGBM on 31 features, threshold 0.70 | dev 0.9476 | yes |
| E005/E006 | second stage with cluster + competition features (sampled population) | +0.0071, but the gain was biased by sampling | replaced by E009 |
| E009 | stage 1 over **all** S1s, full-population competition features | dev 0.9643 | yes |
| E010 | + cross-encoder A (e5-small, folds 1-2) | dev 0.9840 (+0.0197) | yes |
| E011 | two fold-disjoint cross-encoders, stage 2 on all folds | dev 0.9841 | no (no gain) |
| E012 | char 3-gram TF-IDF+SVD retrieval view | recall 0.9740 → 0.9781 | no (small) |
| E013 | + 6 label-free name-rarity features | dev 0.9844 | yes |
| E014 | fine-tuned bi-encoder view | recall 0.9740 → 0.9976 at top-10 | yes |
| E015 | rebuild with the 4th view; stage 1/2 fitted on fold 0 minus dev | dev 0.9898 (stage 1 alone 0.9603); stage 3 no gain | yes |
| E016 | + cross-encoder B (e5-base, 3M pairs, folds 1-4) | dev 0.9905 | yes |
| E017 | + competition features on 5 name similarities | dev 0.9910 | yes |
| E018/E020 | self-training of cross-encoder B on test pseudo-labels | not measurable on dev; LB 0.985645 | yes (confirmed allowed by the organisers) |
| E022 | + Qwen3-4B LoRA reranker on the (0.1, 0.9) band | dev 0.99116 (+0.00019), OOF +0.00030; band AUC 0.922 vs cross-encoder 0.770 | yes |
| E024 | + normalisation v2 features | dev 0.99098 (neutral; aimed at France) | yes |
| E025 | test-like competitor density (keep 0.78) | test-like-density dev 0.98975 → 0.99039 | yes |
| E026 | LightGBM tuning (lr, leaves, min leaf, feature fraction, L2, 3 seeds) | all within ±0.00015 | no (saturated) |
| E027 | + multilingual-e5-large cross-encoder (1.5M pairs, folds 1-4), scored on final candidates only | dev 0.99104 (+0.00004) | base for E029/E030 |
| E028 | importance-weighted stage 2 for the unseen country (domain AUC 0.919) | no gain | no |
| E029 | e5-large continued on francized train pairs (hand-written generic-word dictionary) | francized-dev AUC 0.911 -> 0.992 | base for E030 |
| E030 | self-training of E029 on 350k confident test pseudo-labels (France, US, India) | French uncertain pairs/S1 0.331 (lowest) | yes |
| E032/E033 | stricter candidate filter (3.83/S1) refits of E023b-ST / E030 | dev 0.99095 / 0.99099; LB 0.987692 | yes |
| E034 | + listwise owner model OW04 (owner probability + margin) | dev **0.99154** (+0.00055), OOF 0.99142; LB **0.98793** | **final** |
| E035 | + a second e5-large pair cross-encoder | dev 0.99153 (flat) | no |
| E036 | France self-training round 2 (teacher E034, 450k French pseudo-labels) | French uncertain pairs/S1 0.332 -> 0.288; US/India unchanged | no (see E037) |
| E037 | + French candidate rescue (France-adapted CE >= 0.5 at the filter) | +1,527 French matches; LB 0.987876 (flat) | no |

#### B.2 Stage-2 features (80)
- **Stage 1 (32 + probability):** `cos_{name,addr,both,both_ft}`; `name_{ratio,tset,tsort,partial,jw,full_tset,exact_core,len_l,len_r}`;
  `addr_{tset,partial,tsort,empty_r}`; `num_{jaccard,common,first_eq}`, `post_eq`;
  `ctx_{rank_in_s1,gap_to_best_s1,n_cands_s1,rank_in_cand,gap_to_best_cand,n_s1_for_cand}`; `is_s3`;
  `rank_{name,addr,both,both_ft}`; `prob`.
- **Scores (3):** cross-encoder A, cross-encoder B, LLM P(Yes) (missing outside the band).
- **Cluster (21):** stage-1-probability aggregates per S1 and record (`s2_prob`, rank, gap to max, counts ≥ 0.5 / 0.9,
  sum, best other S1, margin) and sibling similarity to the S1's confident candidates (max and weighted mean of 6
  similarities, anchor count).
- **Competition (12):** best other-S1 value and margin for `cos_name`, `name_ratio`, `name_jw`, `name_full_tset`,
  `name_tsort`, and v2 `name2_ratio`.
- **Normalisation v2 (5):** `name2_{ratio,tset,tsort}`, `addr2_{tset,tsort}`.
- **Rarity (6):** same-name counts among S1s and among records, for both sides; rarest-token document frequency.

#### B.3 Hyperparameters and compute (RTX 5080 16 GB, bf16; 64 GB RAM; 32 threads)
| Component | Settings | Time |
|---|---|---|
| e5-small embeddings | 3 views × train + test, max length per view, float16 output | ~75 min |
| Bi-encoder (E014) | e5-small, 1.5M pairs, batch 256, lr 3e-5, InfoNCE scale 20, max len 96, 1 pass | 18 min + 10 min embedding |
| Stage 1 (E015) | LightGBM binary, lr 0.1, 127 leaves, min leaf 100, feature/bagging fraction 0.8, L2 1, ≤1000 rounds, early stop 50; 200k S1, 4 groups | ~3.4 h incl. retrieval and scoring of 3.9M S1 |
| Cross-encoder A (E008) | e5-small, 2M pairs (40% pos), batch 128, lr 3e-5, warm-up 6%, 1 epoch, max len 96 | 24 min + 1.4 h scoring |
| Cross-encoder B (E016) | e5-base, 3M pairs, batch 64, lr 2e-5, 1 epoch | 2.4 h + 1.2 h scoring |
| LLM reranker (E022) | Qwen3-4B bf16, LoRA r 16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down; lr 1e-4 one-cycle; batch 16; max len 256; 40k prompts, 1 epoch | 1.1 h train; 3.3 h scoring (batch 32) |
| Cross-encoder C (E027) | multilingual-e5-large, 1.5M pairs, batch 32, lr 1.5e-5, 1 epoch | 2.2 h + 1.8 h scoring |
| E029 / E030 | continued training, lr 1e-5: 450k francized+original pairs / 465k pseudo-labelled+original pairs | 1 h / 55 min + 20 min scoring each |
| Stage 2 (final) | same LightGBM settings; fitted on fold 0 minus dev; competition density keep 0.78 | ~0.5-1.5 h CPU |
| Total | | ~36 h wall clock, ~34 GPU-hours |

#### B.4 Error analysis and probes
- **Before the bi-encoder (E010):** 51% of the dev loss was never-retrieved pairs, 3x worse in India. The bi-encoder
  recovered 90.7% of them.
- **After (E015/E017):** ~85% of the dev loss is retrieved-but-rejected true matches, mostly empty-address records
  whose name nearly matches several S1s. On this band the reranker separates much better than cross-encoder B
  (AUC 0.922 vs 0.770; stage 2 as a whole 0.939).
- **France (public LB, on sub-03):** emptying all French rows gave 0.821, implying French F0.5 ≈ 0.89 against ≈ 0.957
  for US + India. A stricter French threshold (0.95) gained +0.0022, so the French errors were false merges.
  Cross-encoder A alone brought French mean matches per S1 from 3.375 to 3.182, in line with US and India.
- **Dev by country:** E022 US 0.99047, India 0.99220. India overtook US once the bi-encoder fixed Indic-script
  retrieval.

#### B.5 Models and licences
| Model | Licence | Parameters | Use |
|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 118M | retrieval embeddings; bi-encoder and cross-encoder A (fine-tuned on train only) |
| intfloat/multilingual-e5-base | MIT | 278M | cross-encoder B (fine-tuned on train only) |
| intfloat/multilingual-e5-large | MIT | 560M | cross-encoder C (E027/E029/E030) and owner model OW04 (fine-tuned on train only; self-training on test inputs, allowed by the organisers) |
| Qwen/Qwen3-4B | Apache-2.0 | 4.0B | LoRA reranker (adapters trained on train only) |
| LightGBM | MIT | n/a | stage-1 and stage-2 classifiers |

All models run offline and are ≤ 8B parameters. We use no external data, APIs, registries, geocoding or hosted LLMs.
Normalisation uses small hand-written generic dictionaries (legal-form words, street-type abbreviations, `&`,
`null`). `country` is an open set of strings; nothing branches on a specific value.

#### B.6 Submission history (public LB)
| Tag | Experiment | Dev / OOF F0.5 | Public LB |
|---|---|---|---|
| sub-01 | all-empty format probe | n/a | probe |
| sub-02 | E004 baseline | 0.9476 / 0.9487 | failed at portal evaluation (file passes both validators) |
| sub-03 | E009 full-population stage 2 | 0.9643 / 0.9618 | 0.947598 |
| sub-04 | probe: sub-03 with France emptied | probe | 0.821 |
| (untagged) | probe: sub-03 with French threshold 0.95 | probe | 0.949822 |
| sub-06 | E010 + cross-encoder A | 0.9840 / 0.9838 | 0.975154 |
| sub-07 | E013 rarity | 0.9844 / 0.9843 | 0.975726 |
| sub-08 | E015 bi-encoder view | 0.9898 / 0.9896 | 0.983322 |
| sub-09 | E016 cross-encoder B | 0.9905 / 0.9901 | 0.984727 |
| sub-10 | E020 (E017 + self-training) | 0.9910 / 0.9907 | 0.985645 |
| sub-11 | E017 (sub-10 without self-training; A/B) | 0.9910 / 0.9907 | 0.985144 |
| sub-12 | E023b-full (reranker full band, density, norm2, 4.74/S1 filter) | 0.99100 / 0.99096 | 0.98631 |
| sub-13 | sub-12 probabilities, France threshold 0.55 | same | 0.985959 |
| sub-14 | sub-12 probabilities, France threshold 0.90 | same | 0.986359 |
| sub-15 | E023b-ST, France 0.85 | 0.99100 / 0.99096 | not uploaded (team slots) |
| sub-16 | E033: E030 self-trained e5-large, 3.98 cands/S1, France 0.85 | 0.99099 / 0.99100 | 0.987692 |
| sub-17 | E034: E033 + owner model OW04 | 0.99154 / 0.99142 | 0.98793 |
| sub-18 | E037: E034 + France self-training round 2 + French candidate rescue | 0.99154 / 0.99142 | 0.987876 |
| sub-19 | **E034 final** (re-upload of sub-17's file) | 0.99154 / 0.99142 | 0.98793 |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
