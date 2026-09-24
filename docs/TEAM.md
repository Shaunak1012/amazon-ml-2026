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

## Branches
- `main`: always runnable; tests green. Only merge via PR or a fast-forward after tests pass.
- Work branches: `<name>/<topic>` e.g. `asha/deberta-ft`, `ravi/img-embeds`. Short-lived (< 1 day).
- Experiment code that won't be reused can stay on its branch; the **OOF files** are what matter.
- Rebase your own branch on `main` freely; **never force-push shared branches or `main`**.
- Submission tags `sub-NN` and the final tag `final` are created only by the Lead from `main`.

## Sharing OOF predictions
- Everyone writes via `src.oof.save_oof(...)` → `OOF_DIR/<exp_id>/{oof.parquet,test.parquet,meta.json}`.
- **Same folds file** for all (`DATA_DIR/folds_s42_k5.csv`, made once by Lead). `stack_oofs` refuses mismatched folds.
- OOF preds in the **original target space**; ids as strings exactly as in the data.
- Shared storage (pick one on Day 1, record in DECISIONS.md): a shared Google Drive/OneDrive folder synced to
  `OOF_DIR`, or an S3 bucket (`aws s3 sync oof/ s3://<bucket>/oof/`). OOF files are small (MBs); never commit them.
- Add the EXPERIMENTS.md row when you upload OOFs, so the Lead knows it's ready for blending.

## Communication
- Discord channel: monitor alerts + short updates. Decisions go in `docs/DECISIONS.md`, not only chat.
- End of each block (~every 6–8 h): 3-line DAILY_LOG.md entry per person.
