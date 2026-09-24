# Decision log

Newest first. Each entry: what we chose, what else we considered, why — in plain language.

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
