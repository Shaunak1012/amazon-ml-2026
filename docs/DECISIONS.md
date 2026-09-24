# Decision log

Newest first. Each entry: what we chose, what else we considered, why — in plain language.

## 2026-09-25 — EDA-driven design decisions (P0) — full report: `python scripts/eda.py` → runs/eda/report.md
Facts (train): singleton rate **5.6%**; mean **3.46 matches/S1** (80% of S1 have both S2 and S3 matches);
**0 records matched to >1 S1**; **0 cross-country pairs**; ~26% of S2/S3 records are distractors (no S1);
India S2/S3 names are **~24% non-Latin** (Devanagari **and** other Indic scripts: Kannada, Telugu, Gujarati),
Indic scripts also appear in addresses (state names); some matched names are **replaced by gibberish**
("Nylaorbiquo") while the address is intact; "F/K/A" names; leetspeak ("C0mpany"); literal `null` in addresses;
34% of test S1 names occur verbatim as train S1 names (names are reused across different businesses).
No row-order or ID-number relationship between S1 and its matches (Spearman ≈ 0.0001) → no leakage.
Test differs: France 15% of S1; **(S2+S3)/S1 = 5.75 in test vs 4.68 in train** → more records per entity or more
distractors in test.

Decisions:
1. **Hard assignment constraint** — each S2/S3 record goes to at most one S1 (0 violations in 7.6M pairs).
2. **Block within the same `country` string** — 0 cross-country pairs; generic (works for France, no hard-coding).
3. **Address is a first-class signal**: blocking and features get an address view (house numbers, street tokens,
   locality), not just names — names can be gibberish or shared by different businesses.
4. **Transliterate every non-Latin script** (anyascii) for char features; keep original text for multilingual e5.
   Clean literal `null` tokens.
5. **Empty-set decisions are rare** (5.6%), so recall matters more than the metric name suggests: most entities need
   *some* match; the expected-F0.5 decision layer handles the trade-off per entity.
6. **Validation:** 5 folds by S1 (random, seed 42; S1 entities are independent). Candidates for a held-out fold are
   retrieved from the **full** train S2/S3 pool (same pool size as test). Dev loop = 100k S1 sampled from fold 0.
   Plus leave-one-country-out (US↔India) for the France margin.
7. **Probe #1 (all-empty)** is informative: its public score = public singleton rate; compare to train 5.6% to
   see whether test's higher record ratio means more distractors or more matches per entity.

## 2026-09-24 — Initial strategy for Business Entity Resolution (before seeing the data; revisit after EDA)
- **Chose:** a classic, auditable three-stage pipeline.
  1. **Normalise** names and addresses: lowercase, strip accents, unify punctuation and `&`, map legal suffixes to
     one form and split them off (a "core name"), expand street abbreviations, pull out numbers and postal codes.
  2. **Block** (candidate generation) with a union of cheap high-recall retrievers: character n-gram TF-IDF on
     names, TF-IDF on name + address, multilingual sentence embeddings (top-K on GPU), and exact keys (postal code +
     first name token). Measure **blocking recall vs. candidates per entity** on train and pick the smallest K that
     keeps recall ≈ 99%.
  3. **Match** with a LightGBM pair classifier on similarity features (rapidfuzz ratios, TF-IDF and embedding
     cosines, number and postal-code agreement, rank and gap-to-best within each S1's candidates), then a
     **decision layer built for macro F0.5**: calibrated probabilities, a threshold tuned on out-of-fold
     predictions, "each S2/S3 record belongs to at most one S1" conflict resolution, and choosing the empty list
     when no candidate is confident (singletons score a full 1.0).
- **Validation:** GroupKFold by S1 entity (all candidates of an entity in the same fold). Also **leave-one-country-out**
  (train on US, test on India, and the reverse) as a stand-in for the unseen **France** test country.
- **Alternatives considered:**
  - Fine-tuned cross-encoder or LLM matcher as the main model: slower, harder to audit; kept as a Day-2 extra feature
    if the GBDT plateaus.
  - Pure embedding similarity + threshold: weak on typos and numbers, and poor precision.
  - Geocoding or business lookups: **banned** (disqualification).
- **Why:** F0.5 rewards precision and singletons, so an explicit, well-calibrated decision layer matters as much as
  the model. GBDT on string features is fast to iterate (minutes per run on 32 CPU threads), easy to explain to
  Amazon scientists, and uses only MIT/Apache components. The country-agnostic features and leave-one-country-out
  checks target the France shift directly.
- **Confidence:** medium. Blocking recall and the singleton rate from EDA may change the balance.

## 2026-09-24 — Repo scaffold before data release
- **Chose:** a problem-agnostic scaffold (metrics registry, shared folds, OOF storage, submission validator,
  heartbeat monitor with Discord alerts) and written playbooks, instead of pre-building a solution.
- **Alternatives:** guessing the task (e.g. re-building the 2025 price model) — rejected: wasted effort if the task changes.
- **Why:** the first 3 hours decide how fast we iterate for the next 69; tooling that prevents format errors
  and silent crashes is useful for any task.

## 2026-09-24 — Environment on this Windows machine
- **Chose:** Python 3.11 venv; pinned pandas 2.3.3 and torch 2.8.0+cu128 — both load under Smart App Control
  (torch 2.11 and pandas 3.0 were blocked). Verified sm_120, bf16, matmul. WSL2 recommended but optional now.
- **Alternatives:** latest wheels (blocked by Smart App Control); turning SAC off. That is reversible on this build
  (post-April-2026 updates), so it's the fallback if a needed package is ever blocked.
- **Why:** get a working CUDA stack without changing security settings.
