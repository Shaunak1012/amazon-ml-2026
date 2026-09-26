# Team workflow

## Roles (3–4 people; rotate if someone is blocked)
| Role | Owns | Day-1 focus |
|---|---|---|
| **Lead / validation** | rules, metric, CV, submissions ledger, final ensemble, approach doc | COMPETITION.md, metric tests, folds, submit decisions |
| **Modeling A (text/NLP)** | text encoders, fine-tunes, text features | TF-IDF + embedding baselines → DeBERTa/e5 fine-tunes |
| **Modeling B (vision/multimodal)** | image download/cache, image embeddings, fusion models | robust image downloader, CLIP/SigLIP embeddings |
| **Features / GBDT / infra** (4th person) | tabular + regex features, GBDT stack, EDA, monitoring, AWS | EDA notebook, GBDT baseline, keeps GPU queue full |
With 3 people, merge the 4th role into Lead + Modeling A.

The **local RTX 5080 is shared**: one queue owner (see `docs/DAILY_LOG.md` "GPU queue"), everyone else uses CPU,
AWS, or Kaggle/Colab notebooks for their own experiments.

## Branches (feature-branch workflow since 2026-09-26)
- **Nobody commits directly to `main`.** Each teammate works on their own branch: `shreyas`, `shaunak`, `solanki`,
  `shivam` (all created from `main` at `aca60ee`, tracked on origin).
- `main` stays runnable with tests green; changes reach it **only through a reviewed PR** from a personal branch.
- Keep your branch current with `git fetch && git merge origin/main` (or rebase your own branch); **never
  force-push `main` or someone else's branch**.
- Submission tags `sub-NN` go on the commit that produced the uploaded file (on its author's branch) and are logged in
  `docs/SUBMISSIONS.md`. Submissions are shared (5/day for the team): claim a slot in the ledger before uploading.
- Large artefacts (data/, runs/, checkpoints/, oof/, submissions/) are never committed; share them out of band.

## Sharing OOF predictions
- Everyone writes via `src.oof.save_oof(...)` → `OOF_DIR/<exp_id>/{oof.parquet,test.parquet,meta.json}`.
- **Same folds file** for all: `data/cache/folds_s1_k5.parquet` (written by `python scripts/make_folds.py`: 5 folds over
  train S1, seed 42, plus a fixed 100k-S1 `dev` subset of fold 0). `stack_oofs` refuses mismatched folds.
- OOF preds in the **original target space**; ids as strings exactly as in the data.
- Shared storage (pick one on Day 1, record in DECISIONS.md): a shared Google Drive/OneDrive folder synced to
  `OOF_DIR`, or an S3 bucket (`aws s3 sync oof/ s3://<bucket>/oof/`). OOF files are small (MBs); never commit them.
- Add the EXPERIMENTS.md row when you upload OOFs, so the Lead knows it's ready for blending.

## Submission queue (5 per day for the whole team, unused slots are lost at the daily reset)
One queue, one bar, one submitter. It doesn't matter whose model it is; the best validated one goes in.
1. **Same yardstick:** score on the shared folds (`data/cache/folds_s1_k5.parquet`) with `er_f05`
   (`src/metrics.py`). Numbers from other splits or other metric code aren't comparable and don't count.
2. **Entry ticket:** validation F0.5 reported (mean ± std over folds, or the dev subset for quick checks); both
   validators pass (`python -m src.er_submission validate ...` and `data/utils/validate_submission.py`); **keep the
   exact code** that produced the file. Teammates may work outside this repo. But if a teammate's submission becomes
   our best or final one, its code must be added to the repo, because the final zip must reproduce it (the rules tie
   the artefacts to "the best solution submitted").
3. **Bar:** it beats our best *submitted* validation score by more than fold noise, **or** it answers a specific
   question (a probe). Ties go to the simpler or faster pipeline.
4. **Post in the team channel:** `SUBMIT REQUEST: <exp id> | val F0.5 <x ± s> | branch <name> | why`. The lead
   checks the gate, runs `scripts/tag_submission.py` (tag `sub-NN` + ledger row), and the **designated submitter**
   uploads. Nobody uploads on their own, since one stray upload burns a team slot.
5. **Blend before you compete:** if your model scores the same candidate pairs, save its pair probabilities with
   `src.oof.save_oof`. A blend often beats both models, and then it's a joint submission.

Default split (the lead can reallocate):
| Day | Slots |
|---|---|
| 25 Sep | probe · baseline · 2 flex (any teammate's model that passes the gate) · 1 spare |
| 26 Sep | France probe · 4 flex |
| 27 Sep | final candidates (by ~18:00 IST) · ≥2 held for fixes only |

## Communication
- Discord channel: monitor alerts + short updates. Decisions go in `docs/DECISIONS.md`, not only chat.
- End of each block (~every 6–8 h): 3-line DAILY_LOG.md entry per person.
