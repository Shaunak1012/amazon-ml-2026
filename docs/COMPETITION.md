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
- Row counts, sizes and sha256 per file: **TBD when the data arrives**. Record them here.

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
- **Models:** the final model must be **MIT or Apache-2.0 licensed** and **≤ 8B parameters**.
- **🚫 No external data lookup.** No entity-resolution APIs, no government/business registries, **no geocoding
  APIs**, no internet data augmentation. Evidence of this means **immediate disqualification**. Use only the
  provided training data. (Pretrained open models within the licence and size limits are fine. Hand-written
  normalisation rules, e.g. abbreviation maps, are our own code, not external data. We're fairly confident that's
  allowed but haven't confirmed it with the organisers.)
- One login per participant; desktop or laptop only; simultaneous logins can terminate the session.
- Multiple IDs, cheating or plagiarism mean disqualification.

## Submission limits and leaderboard
- **Max 5 submissions per day**, for 3 days (15 total); the submit button is then disabled. Daily reset time is not
  stated (assume 00:00 IST).
- **Public LB = a subset of test; private LB = the rest.** The final ranking uses the **private** LB (the statement);
  the guidelines say shortlisting uses both leaderboards. Always submit predictions for the full test set.
- Keep a version history of every submission (our `sub-NN` tags plus `docs/SUBMISSIONS.md`).

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
