# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** SHSHSHSH  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
*Living draft, updated with every leaderboard submission. Current submission: see the version history table below.*

Planned approach: a retrieve → score → decide entity-resolution pipeline. Normalisation with transliteration of Indic scripts; blocking within country using multilingual embeddings of name and address views, character n-grams and exact keys; a LightGBM pair scorer on string, number and context features; and a decision layer tuned for per-entity F0.5 with a one-Source-1-per-record assignment constraint.

---

## 2. Methodology

### 2.1 Problem Analysis
- Train: 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3 records; 7,638,365 match pairs; 5.6% of S1 have no match; mean 3.46 matches per S1.
- No S2/S3 record matches more than one S1; no match crosses countries.
- ~24% of India S2/S3 names are in Indic scripts (Devanagari, Kannada, Telugu, Gujarati); some matched names are replaced by unrelated strings while the address is intact, so addresses are a primary signal.
- Test adds France (15% of S1, unseen in train) and has more S2+S3 records per S1 (5.75 vs 4.68).

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]
- **Candidate pairs generated:** [total]
- **How you ensured true matches were not lost:**

---

## 4. Matching Model

**Features used:**
- Name features: [e.g., Jaccard, Levenshtein, phonetic encoding]
- Address features: [e.g., token overlap, edit distance, PIN code matching]
- Other: []

**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc.]  
**Threshold selection method:** [e.g., F_0.5 optimization on validation set]

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Version history (leaderboard submissions)

| Tag | What | Validation F0.5 | Public LB |
|---|---|---|---|
| sub-01 | All-empty format probe (no model) | n/a | pending |
| sub-02 | Baseline: multilingual-e5-small blocking (name, address, name+address views; top-10 each from S2 and S3, same country) + LightGBM on 31 similarity/context features + threshold 0.70 with one-S1-per-record assignment | 0.9476 (dev, 100k held-out S1); 0.9487 (5-fold OOF) | pending |
| sub-03 (LB 0.947598) | Stage 1 over every S1 (train out-of-fold), top-15 candidates, second-stage LightGBM with cluster and full-population competition features, expected-F0.5 decisions | 0.9643 (dev); 0.9618 (OOF) | pending |
| sub-06 (LB 0.975154) | sub-03 pipeline + cross-encoder (multilingual-e5-small fine-tuned as a sequence-pair classifier on 2M stage-1 candidate pairs from folds 1-2) as a stage-2 feature | 0.9840 (dev); 0.9838 (OOF) | pending |
| sub-08 | sub-06 + label-free name-rarity features (same-name counts per country, rarest-token frequency) targeting empty-address candidates | 0.9844 (dev); 0.9843 (OOF) | pending |

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
