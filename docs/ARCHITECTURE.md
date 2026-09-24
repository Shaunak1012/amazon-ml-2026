# Architecture & roadmap: Business Entity Resolution

Status: **proposal v1** (2026-09-25, before EDA). Items marked *(EDA)* get confirmed or changed once we've
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
  Extract house numbers, postcode/PIN, city and landmark phrases.
- Pure functions, no country-specific branching (contract in WORKERS.md). **Cloud-able.**

### [B] Blocking: maximise recall at a bounded candidate count
Union of complementary retrievers, each giving top-K:
1. **Dense (main retriever).** Embed `core name | city` (and a name-only view) with **multilingual-e5**
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
2. **Assignment constraint:** give each S2/S3 record only to its highest-probability S1 *(if EDA confirms records
   never map to more than one S1)*.
3. **Per-S1 expected-F0.5-optimal set:** sort candidates by p and evaluate the expected F0.5 of each prefix,
   including the **empty set** (the singleton decision). Choose the best. This is theory-backed (e.g. Jansche 2007;
   Ye et al. 2012), cheap, and usually worth points over a global threshold. **Cloud-able** (pure maths, easy to
   unit-test).
4. **France / unseen-country safety:** use leave-one-country-out CV to measure how calibration drifts on an unseen
   country, and apply a correspondingly more conservative margin to country strings not seen in training.

### Validation
- Hold out ~10–15% of train S1 (GroupKFold by S1), but **retrieve from the full train S2/S3 pool**. That mimics
  test difficulty (the pools are the same size). Score with the exact `er_f05`.
- **Leave-one-country-out** (US→India, India→US) as the stand-in for France.
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

## 3. Roadmap (window 25 Sep 00:00 → **27 Sep 23:59 IST**; H0 = 00:00, 25 Sep)
| When | Work | Deliverable / gate | Submissions |
|---|---|---|---|
| **H0–4** | EDA (singleton rate, matches per S1, one-S1-per-record check, cross-country, scripts, address formats); validation split; normalisation v0 | DECISIONS entry on validation; EDA facts in COMPETITION.md | – |
| **H4–9** | Embed all records (GPU, background); dense blocking + recall curve; **baseline**: embedding similarity + tuned threshold + decision layer v0 | first CV score; **sub-01** (early LB anchor, secures a top-500 place for the credits) | 1–2 |
| **H9–24** | Sparse + key blockers, union, K choice; features v1; LightGBM + OOF; expected-F decision | CV v1 ≫ baseline; **sub-02/03** | 2 (Day 1 total ≤ 4) |
| **H24–44** | Error analysis; fine-tuned bi-encoder (recall) and/or cross-encoder (precision); LOCO for France; feature v2 | each gain > fold std gets a submission | 3–4 (Day 2) |
| **H44–60** | Final features, seed/fold bagging, calibration, unseen-country margin; full-train refit; timing test on full test | frozen pipeline; REPRODUCE.md filled | 2–3 |
| **H60–68** | **Code freeze.** Reproduce from a clean clone; zip; methodology doc; final pick on CV | **final submission by ~H66 (27 Sep ~18:00 IST)** | ≥ 2 reserved |
| H68–72 | Buffer only (portal issues). No new ideas. | zip uploaded | – |

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
