# Submission checklist

## Before EVERY leaderboard submission
- [ ] **Worth it?** CV beats best submitted CV by more than fold std (or tests a CV↔LB question we need answered).
- [ ] **Budget:** remaining today (SUBMISSIONS.md) > reserve; ≥ 2 kept for the final day.
- [ ] **Test preds come from the same pipeline as the OOF** (same features, transforms, inverse transform, clipping).
- [ ] **Validator passes:** `python -m src.submission validate <file> --sample <official sample>`
  - columns + order exactly as sample; header present; no index column
  - row count = test rows; id set identical; id order as sample (unless grader joins by id)
  - ids as strings (leading zeros preserved); no duplicates
  - no NaN / empty / inf; values in valid range (e.g. non-negative prices → `--nonneg`); correct units/format
  - not constant; distribution of test preds ≈ distribution of OOF preds (median, min/max)
- [ ] Code + config committed; `python scripts/tag_submission.py ...` → tag `sub-NN`, ledger row; push tag.
- [ ] After upload: fill public LB in SUBMISSIONS.md + EXPERIMENTS.md; note CV↔LB gap.

## Final submission (by H68)
- [ ] Final pick decided on CV (and LB agreement), recorded in DECISIONS.md.
- [ ] Reproduced from a clean clone (teammate or cloud session) → same predictions (max abs diff ≈ 0).
- [ ] `python scripts/make_submission_zip.py --predictions <final.csv> --sample <sample> --doc <approach.pdf>`
- [ ] README best CV/LB updated; tag `final`; pushed.
- [ ] Approach doc uploaded; zip uploaded; confirmation screenshots saved.

---
## Approach document template (1–2 pages)

**Team:** <name> · **Members:** <aliases> · **Final public LB:** <score> · **Local CV:** <score ± std>

### 1. Problem framing
What is predicted, the metric and why it matters for modeling choices (e.g. SMAPE → relative error →
log-target). Key data properties found in EDA (size, skew, missing modalities, shift).

### 2. Validation
Fold scheme and why it mirrors the test split (groups/stratification). Leakage checks performed.
CV↔LB correlation across our submissions (small table or one sentence).

### 3. Models
Each ensemble component: backbone, inputs, loss, key hyper-parameters, training time on RTX 5080.
Pretrained models used + licence/size compliance.

### 4. Features & preprocessing
Text cleaning, extracted structured fields (regex/units), image preprocessing, tabular features.

### 5. Ensembling
Blend/stack method, weights, how selected (OOF), stability across folds.

### 6. Results
| Model | CV | Public LB |
|---|---|---|
| Baseline | | |
| … | | |
| **Final ensemble** | | |

### 7. What didn't work
Short bullets with the measured effect (e.g. "full fine-tune of X: +0.0 CV, 5× cost").

### 8. Reproducibility
Hardware, runtime, `README.md` commands, git tag of the final submission.
