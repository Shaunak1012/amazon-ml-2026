# Competition facts: Amazon ML Challenge 2026 (Unstop, India)

> Source of truth: the organiser PDFs in `ProblemStatement&More/` (gitignored; the problem statement and
> the guidelines). Filled in 2026-09-24. Where the two PDFs disagree, both versions are recorded below.

## The task: Business Entity Resolution
We get business records from **3 independent sources** with no shared identifiers. **Source 1 is the deduplicated
reference.** For **every Source 1 entity**, predict the set of **Source 2 and Source 3 records** that refer to the
same real-world business. Each S1 entity can match **zero, one or many** S2/S3 records. Records are text only
(name, address, country); there are **no images**.

Noise to expect (from the statement):
- **Names:** abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal-suffix differences, DBA/trade
  names, `&` vs "and", word-order swaps, typos, transliterations.
- **Addresses:** abbreviations (Rd/Road, St/Street), transliteration variants, missing parts (no PIN code, no
  state), landmark references ("Near SBI ATM"), municipal numbering formats, reordered parts.

**⚠️ Distribution shift:** train covers **US and India** only. **Test also contains France**, which never appears in
training. The rules say to treat `country` as an open set of strings: never hard-code, filter or one-hot on
{US, India}. Every test entity, including the French ones, must appear in the submission.

## Metric (exact)
F_β with **β = 0.5** (precision counts 2× as much as recall), **macro-averaged per Source 1 entity**:
- For each S1 entity: P = |pred ∩ true| / |pred|, R = |pred ∩ true| / |true|,
  F_0.5 = (1.25 · P · R) / (0.25 · P + R).
- **Singletons count.** No true matches + empty prediction = **1.0**; no true matches + any prediction = **0.0**.
- Average over **all** S1 entities in the evaluation set.
- Worked example from the statement: pred [S2-00047, S2-00193, S3-00812], true [S2-00047, S3-00812] gives
  P = 2/3, R = 1, F_0.5 = **0.714**.
- The statement doesn't define these cases; assumed: true non-empty + empty prediction = 0; no overlap = 0.

Implemented as `src/metrics.py::er_fbeta_macro` (registered as `er_f05`), tested in `tests/test_metrics.py`.

## Data
All files are **tab-separated** (`sep="\t"`); addresses and ID lists contain commas.

| File | Contents |
|---|---|
| `dataset/train/train_source{1,2,3}.tsv` | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/train/train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs; empty = singleton) |
| `dataset/test/test_source{1,2,3}.tsv` | same columns as train, no labels |

- ID prefix gives the source: `S1-`, `S2-`, `S3-` (example format `S1-00001`). There is no separate source column.
- Organiser helpers in `student_resource/`: `utils/validate_submission.py` (stdlib-only format checker; doesn't
  score) and `Documentation_template.md`.
Received 2026-09-25 00:22 IST. Our copy is at `data/dataset/`, `data/utils/`, `data/Documentation_template.md`.
Row counts exclude the header and come from `wc -l`; re-check them with pandas during EDA.

| File | Rows | Size | sha256 (first 12) |
|---|---:|---:|---|
| train_source1.tsv | 2,206,821 | 210 MB | 591af0e1dfeb |
| train_source2.tsv | 5,034,616 | 489 MB | 6336c1a055ee |
| train_source3.tsv | 5,285,603 | 504 MB | 67da22f51518 |
| train_ground_truth.tsv | 2,206,821 | 127 MB | 70bc1d8a16c6 |
| test_source1.tsv | 1,732,544 | 175 MB | 3d4a32c54c2c |
| test_source2.tsv | 4,887,273 | 509 MB | 79d906c7497a |
| test_source3.tsv | 5,082,316 | 506 MB | 850942b11d2a |

First look (the first few rows only; EDA will quantify all of these):
- **Scale:** 2.2M S1 × ~10M S2+S3 in train. Blocking must be scalable (no all-pairs); test is about the same size.
- IDs aren't zero-padded and vary in length (`S1-965667`, `S1-925783039`). Always treat IDs as strings.
- **Mixed scripts:** some S2 names are in **Devanagari** (e.g. a Hindi rendering of "Ram Marketing Private
  Limited"), so matching must cross scripts, not just spellings.
- Junk tokens in names (`--`, `<<`), a legal suffix moved to the front ("LLC Moncada …"), an injected accent
  ("Léarning"), and a website domain used as the name (`wilfordhancock.com`).
- **Empty addresses** occur. Addresses vary in case and order ("IA, Iowa City, 1064 Newton Rd, Unit 11") and spell
  states in full or abbreviated ("Texas" vs "TX").
- Ground-truth lists can be long: 3–5 matches per S1, from both S2 and S3.

## Submission format
Two TSV files, both in `output/`:
1. **`matching_results.tsv`**, the **only file scored**; upload it to the portal. Columns
   `source1_entity_id<TAB>matched_entity_ids`. IDs comma-separated with no quoting or spaces. Empty cell means no match.
2. **`candidate_pairs.tsv`**, the blocking output: the **exact** candidate set the final model ran inference on (the
   last filtering stage). Columns `source1_entity_id<TAB>candidate_entity_ids`. Not scored; the organisers use it to
   check blocking recall and reduction ratio, and to verify the pipeline.

Rejection rules (a failed validation is **not scored**; the portal shows `SCORED` with the F_0.5 when it's accepted):
- **Exactly one row per test S1 entity**; missing entities or duplicate `source1_entity_id` rows are rejected.
- ID lists: **only S2-/S3- IDs that exist in the test set**; self-matches to S1 are rejected; **no duplicates** in a list.
- Matches should be a **subset of the candidates**. The validator warns if not; it signals a pipeline bug.

Our checks: `src/er_submission.py` (ours), then `student_resource/utils/validate_submission.py` (theirs), both before every upload.

## Organiser update (26 Sep): candidate set size counts
- `candidate_pairs.tsv` is part of the final submission and is **reviewed with the code that produces it**.
- Blocking must scale (billions of records): no all-pairs comparison; a small candidate set per Source 1 entity.
- **A smaller candidate set per S1 ranks higher** in the final evaluation, beyond the public/private LB.
- Ours: dense retrieval (4 views x top-10, same country) ~62/S1 -> stage-1 filter -> final set by stage-1 prob
  (`--prune-eps`; 0.003 gives ~6.4/S1 at -0.00007 dev F0.5, 0.001 gives ~7.6/S1 at -0.00003). Exact GPU search today,
  swap for ANN (e.g. HNSW) at billion scale; same-country partitioning is the first cut.

## Final submission package (required from every team)
```
<team_name>_submission.zip
├── output/matching_results.tsv, output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}   # self-contained, reproduces both outputs
└── Documentation_template.md                                             # filled-in methodology (.md or .pdf)
```
The top teams' packages are **reviewed in detail** (reproducibility, blocking audit, fair play, model licences)
before final rankings are confirmed. Code needs proper comments describing the functions.

## Rules
- **Models:** the final model must be **MIT or Apache-2.0 licensed** and **≤ 8B parameters**. Read literally as total
  parameters: Qwen2.5-7B (7.6B) is fine; **Qwen3-8B (8.2B) is avoided**; Qwen2.5-3B is excluded (research licence);
  Llama/Gemma licences are not MIT/Apache.
- **🚫 No external data lookup.** No entity-resolution APIs, no government/business registries, **no geocoding
  APIs**, no internet data augmentation. Evidence of this means **immediate disqualification**. Use only the
  provided training data. (Pretrained open models within the licence and size limits are fine. Hand-written
  normalisation rules, e.g. abbreviation maps, are our own code, not external data. We're fairly confident that's
  allowed but haven't confirmed it with the organisers.)
- One login per participant; desktop or laptop only; simultaneous logins can terminate the session.
- Multiple IDs, cheating or plagiarism mean disqualification.

## Rule clarification from the organisers (received 2026-09-25)
- **Prohibited:** external databases, APIs, geocoding or entity lookup, internet-sourced augmentation, and **packages
  bundling external geo/postal/business data** (e.g. libpostal, geocoders, postal-code or gazetteer datasets).
- **Allowed:** pure-algorithm libraries (RapidFuzz, jellyfish, scikit-learn, LightGBM, pandas), general-language
  pretrained NLP/embedding models within the limits, any algorithm using only the provided records, and **small
  hand-written normalisation dictionaries**.
- **Every model** (embedder, reranker, matcher, preprocessing model) must independently be MIT/Apache-2.0, ≤ 8B
  parameters, run **offline** (no live API calls), and be **fine-tuned only on the provided data**. Licences are checked.
- **Hosted LLM APIs (Claude/Gemini/ChatGPT) are not allowed** in the solution.
- **Self-training on unlabeled test records** (pseudo-labels from our own predictions): the team lead's reading on
  2026-09-26, **not confirmed by the organisers**; the statement's Fair Play line says "using only the provided training
  data". **Organisers confirmed on 27 Sep that self-training and synthetic pairs from the provided records are allowed** (unsupervised stats on test, offline rule-based text libraries, own transliteration also fine; cleanco not allowed; candidate_pairs.tsv = the final candidate set fed to the matching model). Used in the final.

### Our compliance (keep current; mirrored in docs/REPRODUCE.md)
| Component | Type | Licence | Params | Verdict |
|---|---|---|---|---|
| intfloat/multilingual-e5-small | embedding model, offline | MIT | 118M | allowed; fine-tuning (if any) only on provided train data |
| microsoft/mdeberta-v3-base (planned) | cross-encoder backbone, offline | MIT | 278M | allowed |
| intfloat/multilingual-e5-base (E016 cross-encoder) | cross-encoder backbone, offline | MIT | 278M | allowed; fine-tuned only on provided data |
| intfloat/multilingual-e5-large (E027 cross-encoder) | cross-encoder backbone, offline | MIT (model card, checked 27 Sep) | 560M | allowed (<= 8B); fine-tuned only on provided data (folds 1-4) |
| **Qwen/Qwen3-4B** (planned LLM reranker, LoRA) | decoder LLM as a pair classifier, offline | Apache-2.0 (model card, checked 26 Sep) | 4.0B | allowed (<= 8B); fine-tuned only on provided data |
| Qwen/Qwen2.5-7B-Instruct (optional larger reranker) | decoder LLM, offline | Apache-2.0 | 7.6B | allowed (<= 8B) |
| Qwen/Qwen2.5-1.5B-Instruct (fallback reranker) | decoder LLM, offline | Apache-2.0 (model card, checked 26 Sep) | 1.54B | allowed |
| LightGBM | pure-algorithm library | MIT | n/a | allowed |
| rapidfuzz, jellyfish, scikit-learn, pandas, numpy, pyarrow | pure-algorithm libraries | MIT/BSD | n/a | allowed |
| anyascii | Unicode → ASCII character transliteration (no geo/postal/business data) | ISC | n/a | allowed (character mapping, not a gazetteer) |
| legal-suffix list, `&`→"and", `null` cleanup (`src/er_normalize.py`) | small hand-written normalisation dictionary | ours | n/a | explicitly allowed |
| postcode / number extraction | regex on provided records | ours | n/a | allowed |
| network calls in `src/` | none (models load from local cache) | — | — | compliant |
Never add: libpostal, geocoders, postal-code/gazetteer/state-abbreviation datasets, or hosted LLM calls. Abbreviation maps are small hand-written
generic dictionaries (organiser-allowed), applied to every row regardless of country.

## Submission limits and leaderboard
- **Max 5 submissions per day**, for 3 days (15 total); the submit button is then disabled. Daily reset time is not
  stated (assume 00:00 IST).
- **Public LB = a subset of test; private LB = the rest.** The final ranking uses the **private** LB (the statement);
  the guidelines say shortlisting uses both leaderboards. Always submit predictions for the full test set.
- Keep a version history of every submission (our `sub-NN` tags plus `docs/SUBMISSIONS.md`). The guidelines say
  "shortlisting will be based on the submitted solutions", and the final source code may be requested later.
- **Which submission counts (our reading, unconfirmed):** the guidelines tie the artefacts to "the **best solution**
  submitted by the team" and say evaluation is "based on performance **across both leaderboards**". So the best
  submission is probably picked, not the last. Query sent to the organisers; until they answer, **every real submission
  must be one we'd be happy to be ranked on**. Probes (e.g. all-empty) score low, so they won't be taken as "best".

## Deadlines (IST)
| Event | Date/time IST |
|---|---|
| Challenge window opens | 25 Sep 2026 00:00 |
| +$100 credits for the top 500 | 48-hour mark (~27 Sep 00:00) |
| **Challenge window closes** | **27 Sep 2026 23:59** |
| Artefacts (doc + code zip) | for the best submission; exact deadline TBD (assume same window) |
| Top 100 announced, then extra details requested | after the round |
| Top-50 (PPI) results | ~2 Oct 2026 |
| Grand Finale (top 10) | 7 Oct 2026 (virtual) |

## Other facts
- The top 50 teams get Applied Scientist Intern PPIs. The top 10 (leaderboard + document) go to the Grand Finale.
- Prizes: ₹1,00,000, ₹75,000 and ₹50,000; swag for the top 10 and the top 10 women-only teams.
- $200 AWS credits per participant, plus $100 for the top 500 at 48 h. Organisers suggest `us-east-1`.
- Queries during the hackathon: the Google Form linked in the guidelines PDF. Tech issues: support@unstop.com.

## Ambiguities (ask via the Google Form if they matter)
1. **Document length:** the guidelines say a **1–2 page** document; the statement says "no page limit" for
   `Documentation_template.md`. Plan: fill the template fully, keeping the core under ~2 pages, with appendices.
2. **Final ranking:** the statement says it's decided by the **private** LB; the guidelines say performance across
   **both** leaderboards.
3. Empty prediction for an entity that has true matches presumably scores 0 (recall 0). Not stated explicitly.
4. Whether hand-written abbreviation or normalisation dictionaries count as "external data". We assume no; confirm if we rely on them heavily.

## Sources
- Organiser PDFs: problem statement + "Guidelines and Key Instructions" (local, gitignored).
- Unstop page: https://unstop.com/hackathons/amazon-ml-challenge-2026-amazon-1743604
- AWS prep guide (J. Mehrotra, AWS, 21 Sep 2026): https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo
- A Kaggle dataset named "amazon-ml-challenge-2026" predates the release. It's unofficial; don't use it.
