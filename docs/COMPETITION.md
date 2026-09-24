# Competition facts — Amazon ML Challenge 2026 (Unstop, India)

> Fill every **TBD** on Day 1 from the official problem statement/rules. Quote formulas and limits
> verbatim; link the source. If something is ambiguous, write the ambiguity down and ask organisers.

## Known before Day 1 (confirmed 2026-09-24 from the Unstop page and the AWS prep blog; sources at bottom)
- Stage 2 = 72-hour hackathon, **25–27 Sep 2026**. Problem + dataset on Day 1, build and submit until Day 3.
  Exact closing time TBD on Day 1 (27 Sep 23:59 or 28 Sep 00:00 IST?). Registration closed; ~89k registered.
- Deliverables: predictions (a CSV; the grader doesn't need a hosted model), a **1–2 page approach document**, and
  code/scripts/notebooks as a **zip**.
- Live leaderboard with team rankings throughout the round.
- **Top 50 teams get PPIs for Applied Scientist Intern** (results ~**2 Oct 2026**). Top 10 teams (chosen on
  leaderboard **and** approach doc) present at the virtual **Grand Finale on 7 Oct 2026**.
- Prizes: ₹1,00,000 (1st), ₹75,000 (2nd), ₹50,000 (3rd); certificates + swag for the top 10 and top 10 women-only teams.
- Compute credits: **$200 AWS credits per registered participant**; **+$100 for the top 500 teams at the 48-hour
  mark**, so being on the leaderboard early pays. The organisers' prep blog recommends `us-east-1`.
- Team of 2–4 with a team leader; cross-college allowed. Each member needs an AWS Builder Center profile alias.

## Priors from past editions
| Year | Task | Data | Metric | Notes |
|---|---|---|---|---|
| 2025 | Product price prediction from catalog text + image URL (multimodal regression) | 75k train / 75k test | SMAPE | Rules restricted model size and license |
| 2024 | Extract entity values (weight, dimensions, voltage…) from product images | images + entity name | F1 (exact string match on "value unit") | Allowed-units list; format "x.x unit" |

Implication: expect product-catalog data (text + images), possibly model-size/licence limits, and a metric
with edge-case quirks. `src/metrics.py` already has `smape` and `extraction_f1`.

## Day-1 sections (fill in)
### Problem
TBD — one-paragraph restatement in our own words + what exactly is predicted per row.

### Metric (exact formula)
TBD — paste verbatim. Note: direction, zero/empty handling, averaging (macro/micro/per-row), rounding.
Implemented as: `src/metrics.py::<name>`; tests: `tests/test_metrics.py::<test>`.

### Data schema
| File | Rows | Size | sha256 (first 12) | Columns (type) |
|---|---|---|---|---|
| TBD | | | | |

Id column: TBD (format, leading zeros?). Target: TBD (range, units, distribution).
Images: TBD (URLs? count? download rate limits?).

### Submission format
TBD — file type, exact columns + order, header, id order, float precision, allowed values/units.
Sample file: `DATA_DIR/<name>` — validate it with `python -m src.submission validate`.

### Allowed / banned resources
- External data: TBD
- Pretrained models (size limit? licence whitelist? open weights only?): TBD
- Paid APIs / LLM APIs (OpenAI, Bedrock, Claude…): TBD
- Hand-labelling / test-set usage: TBD
- Pseudo-labelling on test: TBD

### Submission limits
- Per day: TBD (daily reset time IST: TBD)
- Total: TBD
- Which submission counts for final ranking (last? selected? best?): TBD
- Public/private LB split: TBD

### Deadlines (IST)
| Event | Date/time IST |
|---|---|
| Round opens | 25 Sep 2026 00:00 |
| +$100 credits for the top 500 | 48-hour mark (~27 Sep 00:00) |
| Submission deadline | TBD (27 Sep, time TBD) |
| Approach doc deadline | TBD |
| Code zip deadline | TBD |
| Top-50 (PPI) results | ~2 Oct 2026 |
| Grand Finale | 7 Oct 2026 (virtual) |

### Open questions for organisers
- TBD

### Sources
- Unstop page: https://unstop.com/hackathons/amazon-ml-challenge-2026-amazon-1743604
- AWS prep guide (J. Mehrotra, AWS, 21 Sep 2026): https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo
- Note: a Kaggle dataset named "amazon-ml-challenge-2026" predates the release. It is unofficial; don't use it.
