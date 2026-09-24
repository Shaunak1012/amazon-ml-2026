# Architecture & roadmap: Business Entity Resolution

Status: **v2.1** (v2 accepted 2026-09-25; EDA confirmations below, details in DECISIONS.md).

**EDA confirmed (P0):** one-S1-per-record holds (0 violations) → hard assignment constraint; 0 cross-country pairs →
block within the `country` string; singletons only 5.6% (mean 3.46 matches/S1) → recall matters; ~24% of India S2/S3
names are in Indic scripts (Devanagari, Kannada, Telugu, Gujarati) → transliterate everything; some matched names are
gibberish while the address is intact → **address view is mandatory in blocking and features**; test has more
S2+S3 per S1 (5.75 vs 4.68).

Original note: Items marked *(EDA)* get confirmed or changed once we've
measured them. Timings are estimates for the RTX 5080 + 32-thread CPU; confidence is noted per claim.

## 0. How we see the problem
- It's really **record assignment**: S1 is deduplicated, so each S2/S3 record should belong to **at most one** S1
  entity *(EDA: verify)*. The structure is a bipartite graph S1 ↔ {S2, S3}, not free-form pairwise matching.
- **Scale:** 2.2M S1 × 10.3M S2+S3 (train); 1.7M × 10.0M (test). All-pairs is about 10¹³ comparisons, so blocking
  is the first engineering problem.
- **The metric shapes the output:** macro F0.5 per S1 entity (precision ≈ 2×), singletons worth 1.0. The final
  lists should be *confident subsets*, with an explicit "no match" decision.
- **Shift:** France is test-only, and there's cross-script text (Devanagari ↔ Latin). Stay country- and
  script-agnostic.

## 1. Proposal #1: four-stage pipeline
```
 S1/S2/S3 TSV ─▶ [A] Normalise ─▶ [B] Blocking (union of retrievers, both directions) ─▶ candidate_pairs.tsv
                                                     │
                                  [C] Pair scorer: LightGBM on ~50 similarity features (+ later: cross-encoder score)
                                                     │  calibrated P(match)
                                  [D] Decision: one-S1-per-record → expected-F0.5-optimal set per S1 ─▶ matching_results.tsv
```

### [A] Normalisation (CPU, minutes)
- Unicode NFKC, casefold, strip accents, and **transliterate non-Latin scripts to Latin** with a rule-based library
  (`anyascii`, ISC licence; code, not external data). The original text is kept for multilingual models.
- Names: drop junk tokens (`--`, `<<`); map `&` to "and"; split off the legal suffix into its own field (Pvt Ltd,
  LLC, Inc, SARL/SAS …) and keep the **core name**; split website domains used as names into tokens.
- Addresses: expand abbreviations. **Learn the abbreviation and state maps from matched pairs in train** (for
  example, which tokens co-occur in matched pairs such as "TX" and "Texas"), which counts as using provided data.
  Extract house numbers, postcode/PIN, city and landmark phrases. **Label-derived maps are learned inside each
  training fold** (never from held-out labels) and refit on all of train for the test run.
- Pure functions, no country-specific branching (contract in WORKERS.md). **Cloud-able.**

### [B] Blocking: maximise recall at a bounded candidate count
Union of complementary retrievers, each giving top-K:
1. **Dense (main retriever).** Embed a name view **and an address view** (EDA: names can be gibberish) with **multilingual-e5**
   (MIT; handles Hindi, French and English). About 22M texts at ~5–10k/s in bf16 takes roughly 1–1.5 h on the 5080
   (estimate, medium confidence). **Exact brute-force kNN on the GPU**: keys fit in VRAM (5M × 384 fp16 ≈ 3.8 GB),
   using chunked matmul + top-k, partitioned by the `country` string. Minutes, no approximation.
2. **Sparse char n-grams.** TF-IDF on transliterated core names (3–4-grams) with sparse top-K. Catches typos and
   transliterations the embeddings miss.
3. **Keys:** postcode + first core-name token, domain-name tokens, phonetic code of the core name + city.
4. **Both directions:** besides "top-K S2/S3 per S1", run "top-k S1 per S2/S3 record" (the assignment view). This
   catches records that rank low in a crowded S1 neighbourhood.
- Union, then cap to **K per S1 per source**, ranked by best retriever rank. Pick K from the **recall curve** on a
  validation fold; target ≥ 98–99% pair recall. Record recall and reduction ratio for the methodology doc.

### [C] Pair scorer: LightGBM (MIT)
- Features (~50), computed in bulk with `rapidfuzz.process.cpdist` (multithreaded):
  - **Name:** ratio, partial ratio, token-set and token-sort ratios and Jaro-Winkler, on raw, normalised, core and
    transliterated text; char TF-IDF cosine; embedding cosine; legal-suffix compatibility; domain match.
  - **Address:** the same similarity family; house-number equal / conflict / missing; postcode equal / conflict /
    missing; city/state agreement; landmark overlap.
  - **Context (strong in ER):** rank and gap-to-best of this candidate within its S1's list; **mutual rank** (rank
    of this S1 among the record's own S1 candidates); number of candidates; **cluster coherence** (similarity of
    this candidate to the S1's other top candidates, since S2 and S3 copies of one business agree with each other).
  - Script flags, missing-field flags, same-country flag. **No country-identity features.**
- Training: all positives plus hard negatives from blocking (subsample easy negatives), GroupKFold by S1, and OOF
  probabilities saved (`src/oof.py`). Tens of millions of rows are fine in 64 GB.

### [D] Decision layer: optimise the metric directly
1. **Calibrate** OOF probabilities (isotonic).
2. **Assignment constraint:** give each S2/S3 record only to its highest-probability S1 (EDA: 0 violations in
   7.64M train pairs). Kept **switchable** and ablated on held-out F0.5.
3. **Per-S1 expected-F0.5-optimal set:** sort candidates by p and evaluate the expected F0.5 of each prefix,
   including the **empty set** (the singleton decision). Choose the best. This is theory-backed (e.g. Jansche 2007;
   Ye et al. 2012). **An experiment, not a given:** compared against a global threshold and a threshold + top-1
   fallback on held-out per-entity F0.5; kept only if it wins by more than fold noise. **Cloud-able** (pure maths).
4. **France / unseen-country safety:** use leave-one-country-out CV to measure how calibration drifts on an unseen
   country, and apply a correspondingly more conservative margin to country strings not seen in training.

### Validation
- Hold out ~10–15% of train S1 (GroupKFold by S1), but **retrieve from the full train S2/S3 pool**. That mimics
  test difficulty (the pools are the same size). Score with the exact `er_f05`.
- **Leave-one-country-out** (US→India, India→US) as a **robustness check** (does the approach lean on known-country
  patterns?). It is **not** a France score; the only direct France signal is the public-LB France probe.
- Log CV and public LB for every submission; the private LB decides, so trust CV.

## 2. Alternatives if part of #1 fails
| Symptom | Plan B | Plan C |
|---|---|---|
| Blocking recall < 97% at usable K | **Fine-tune the bi-encoder** (multilingual-e5, contrastive on train pairs with hard negatives), which is usually the biggest recall jump, especially cross-script | More key blockers (sorted neighbourhood on core name, postcode-only for short names); raise K and filter with a cheap first-stage scorer |
| Blocking too slow or large | Encode names only with e5-small; partition by country + first char | FAISS IVF (approximate) instead of exact search |
| GBDT plateaus | **Cross-encoder reranker** (fine-tuned `mdeberta-v3-base`, MIT) on the top-N candidates, fed back as a feature | LLM judge (≤ 8B, Apache, e.g. Qwen2.5-7B) **only on ambiguous pairs**; slow, so last resort |
| Decision layer gains are small | Graph step: connected components with S2↔S3 edges, split weak bridges (correlation-clustering-style) | Separate "has any match?" classifier per S1 feeding the empty-set decision |
| France looks bad (LOCO drop is large) | Stricter threshold for unseen countries; char-level features weighted up | Transliteration + multilingual embeddings only for unseen scripts/countries |
| Time crunch | **Safe fallback (kept up to date from Day 1):** exact normalised-key matches + high-threshold embedding similarity, precision-first | Submit the best validated earlier version; never gamble the last submissions |

## 3. Roadmap by phase (each phase ends at a gate; see §8 for exactly what to submit when)
| Phase | Work | Gate to leave the phase |
|---|---|---|
| **P0 Foundations** | Load all TSVs once into cached parquet (strings as-is). EDA: singleton rate, matches per S1 by source, **one-S1-per-record check**, S2/S3 duplicates per entity, cross-country pairs, scripts, address formats, ID/order structure (to know it; never to exploit). Validation split + a **fast dev subset**. Build the **all-empty probe** file | EDA facts + validation in DECISIONS.md; probe passes both validators |
| **P1 Baseline end-to-end** | Normalisation v0; zero-shot multilingual-e5 embeddings for all ~22M records (GPU, cached); exact GPU kNN; recall curve; baseline matcher = embedding + fuzzy-name score + tuned threshold; decision v0 (assignment + threshold); full test run | CV computed on held-out S1; outputs pass both validators; runtime known |
| **P2 Core model** | Sparse + key blockers, union, choose K. Features v1, LightGBM with OOF, isotonic calibration, **expected-F0.5 decision**. Start the **bi-encoder fine-tune** on the GPU in the background (the data has ~10M positive pairs) | CV clearly above baseline (> 3× fold std) |
| **P3 Upgrades from error analysis** | The fine-tuned retriever replaces zero-shot (recall); cross-encoder score as a feature (precision); cluster-coherence / S2↔S3 graph features; LOCO to calibrate the unseen-country margin; **France probe** | Each upgrade is kept only if CV gain > fold std |
| **P4 Consolidate** | Seed/fold bagging, final calibration, full-train refit, full test timing run, commented code, REPRODUCE.md, methodology draft | Frozen pipeline reproduces the outputs from a clean clone |
| **P5 Ship** | Final pick on CV (+ LB sanity check), zip, doc, **final submission by 27 Sep ~18:00 IST**; the rest is buffer | Zip + doc uploaded; SUBMISSIONS.md complete |

## 4. Who does what (docs/WORKERS.md, docs/TASKS.md)
- **Local (critical path, needs data/GPU):** EDA, embeddings, blocking + K, training, OOF, thresholds, LOCO,
  submissions, merging.
- **Cloud (parallel, contract-bound, tested on synthetic data):** normalisation rules (T002), feature groups
  (T004), expected-F decision algorithm (T006 core), pipeline CLI (T007), methodology-doc draft.

## 5. Dependencies to add (all MIT/Apache/ISC; confirm before installing)
- Python: `anyascii` (transliteration), `faiss-cpu` (only if Plan C blocking), `jellyfish` (phonetic codes).
  `polars` is optional for faster I/O.
- Models: `intfloat/multilingual-e5-base` (or `-small`), `microsoft/mdeberta-v3-base` (cross-encoder, Plan B).

## 6. Risks
- Most likely failure: **blocking recall** (cross-script and heavy noise), which the bi-encoder fine-tune addresses.
- Biggest silent risk: **validation that doesn't mimic test** (pool size, France). Mitigated by retrieving from the
  full pool and by LOCO.
- Format risk: removed by our validator + the organisers' validator before every upload.

## 7. Critical review of v1 → v2 changes
Is this the best plan? For a 72-hour event at this scale, a retrieve → score → decide pipeline with a
metric-optimal decision layer is the right backbone (high confidence). It's what strong large-scale ER systems use,
and it's auditable, which the organisers care about. The weak spots of v1 and what changed:
1. **It under-used the labels.** Train has ~2.2M S1 entities with ~10M labelled matches, which is plenty to
   *learn* representations. So the **fine-tuned bi-encoder moves from "plan B" into P2/P3 of the main plan** (it's
   the biggest lever for cross-script and heavily noised recall). The cross-encoder becomes a planned P3 upgrade,
   feeding the GBDT, which keeps it fast and explainable.
2. **Duplicates inside S2/S3.** One S1 often has 2+ S2 and 2+ S3 records, so the candidates of one entity form a
   cluster. v2 adds cluster-coherence features and S2↔S3 agreement, cheap and usually strong.
3. **Iteration speed.** v2 caches everything (parquet, embeddings, candidates, features) and uses a dev subset for
   quick experiments, with full runs only at gates. That turns multi-hour loops into minutes.
4. **Submissions carry information.** The daily quota doesn't roll over, so spare slots become deliberate
   **probes** (all-empty → public singleton rate; France on/off → whether our France predictions help). See §8.
5. **Know the data's structure.** EDA checks ID and row-order patterns only to make sure validation isn't fooled.
   Exploiting them would fail the fair-play audit.
Remaining uncertainty: GBDT vs. a pure neural matcher as the main scorer. The hybrid (neural scores as GBDT
features) keeps most of the upside of both.

## 8. Submission plan (15 total, 5/day, unused slots are lost at the daily reset)
Every non-probe submission needs **CV gain > fold std** over our best submitted run, or it must answer a specific
question. Both validators run first. Record every submission in SUBMISSIONS.md with a `sub-NN` tag.

| Day | # | What | Why |
|---|---|---|---|
| 25 Sep | 1 | **All-empty probe** (every S1 row empty) | Checks the portal format end to end; the score **equals the public singleton rate**, which calibrates the empty-set prior vs. train |
| 25 Sep | 2 | **P1 baseline** | First real LB anchor; CV↔LB check; secures a top-500 place at the 48 h credit cut |
| 25 Sep | 3 | P2 LightGBM v1 (if ready before midnight and it passes the gate) | Main-model anchor |
| 26 Sep | 4 | P2 final: full blocking + expected-F decision | |
| 26 Sep | 5 | **France probe**: #4 with every France entity forced to empty | An LB rise means our France matches are net-harmful, so tighten the unseen-country margin |
| 26 Sep | 6–7 | P3 upgrades (fine-tuned retriever, cross-encoder feature) | Only if the gate passes |
| 26 Sep | (8) | spare | Slots don't carry over, so use for a P3 variant if informative |
| 27 Sep | 9–10 | P4 consolidated candidates (bagged, full refit) | Pick the final on CV |
| 27 Sep | 11 | **Final submission by ~18:00 IST** | Last one = the one we want counted (until organisers confirm which submission counts) |
| 27 Sep | 12–13 | **Reserve**, only for fixes (format, bug) | Never for new ideas after 18:00 |
Probes use the public-LB subset only; they inform coarse decisions (priors, France margin), never fine threshold tuning.
