# Competition facts — Amazon ML Challenge 2026 (Unstop, India)

> Fill every **TBD** on Day 1 from the official problem statement/rules. Quote formulas and limits
> verbatim; link the source. If something is ambiguous, write the ambiguity down and ask organisers.

## Known before Day 1
- 72-hour ML hackathon. Problem statement + dataset released Day 1; submission by Day 3.
- Round opens **25 Sep 2026, 00:00 IST**; 72 hours to submit (→ nominally **28 Sep 2026, 00:00 IST** — verify).
- Deliverables: (1) predictions file in required format, (2) 1–2 page approach document, (3) code/notebooks as a zip.
- Live public leaderboard. Top 10 teams (leaderboard + approach doc) → virtual Grand Finale on **7 Oct 2026**,
  presenting to Amazon scientists. Finalists get a shot at Applied Scientist Intern PPIs.
- Team of 3–4. Registration used AWS Builder Center profile aliases; AWS credits/Free Tier may be available (verify).

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
| Submission deadline | TBD |
| Approach doc deadline | TBD |
| Code zip deadline | TBD |
| Grand Finale | 7 Oct 2026 (virtual) |

### Open questions for organisers
- TBD
