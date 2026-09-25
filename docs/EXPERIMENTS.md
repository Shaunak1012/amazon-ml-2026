# Experiment log

Rules: one row per experiment (all folds). ID = `E###-short-name`, config at `configs/E###-short-name.yaml`,
OOF at `OOF_DIR/E###-short-name/`. CV = mean over folds (± std). LB only if submitted. Runtime = wall-clock GPU/CPU.
Commit the row in the same commit as the code/config (`exp: E### <desc> CV <score>`).

| id | date (IST) | owner | model | features / inputs | CV (± std) | LB | runtime | notes |
|---|---|---|---|---|---|---|---|---|
| E000-template | 2026-09-24 | setup | MLP (synthetic) | synthetic | — | — | — | scaffold smoke test only |
| E001-embed-train | 2026-09-25 | local | multilingual-e5-small, bf16 | name / addr / name+addr views, raw text | — | — | ~21k rows/s (name), 17k (addr), 13.5k (both) | embeddings for all 12.5M train records; e5-base is ~2x slower |
| E002-recall-name | 2026-09-25 | local | exact GPU kNN, same country | name view only, dev 100k S1 vs full pool | recall@10 0.592, @50 0.712 (S2+S3) | — | 112 s | name alone is weak: reused and gibberish names, Indic scripts. Address view needed |
| E002-recall-name-addr | 2026-09-25 | local | exact GPU kNN, same country | name + addr views (union), dev 100k S1 | recall@10 0.946, @50 0.975 (addr alone 0.868 / 0.910) | — | ~10 min (GPU shared) | address view carries most signal; union is a workable ceiling for the baseline |
| E003-baseline | 2026-09-25 | local | LightGBM (4 fold models) on 31 features + decision layer | 3 e5-small views, k=10 per view/source (50 pairs/S1, recall 0.974); train 200k S1 folds 1-4, dev 100k S1 fold 0 | **dev F0.5 0.9476** (train OOF 0.9473); US 0.9558, India 0.9351 | — | 26 min | global threshold 0.70 best; assignment no effect at t=0.7; expected-F 0.9462; top-1 fallback 0.9391 (worse). Top features: cos_both, num_jaccard, rank context |
| E004-predict-sub02 | 2026-09-25 | local | E003 recipe, 5 fold models on 300k train S1 (all folds) | 3 e5-small views, k=10 (86.7M test pairs) | train OOF F0.5 0.9487 (thr 0.70) | pending | 67 min | test: 94.1% non-empty, 3.20 matches/S1; France 5.0% empty / 3.28 (US 6.0% / 3.22, India 6.2% / 3.16). First run OOM-thrashed (116 GB commit): fixed by chunked cosines + memory-mapped pool |
| E005-stage2 | 2026-09-25 | local | E003 + second-stage LightGBM on 31 stage-1 feats + 19 cluster feats (built from stage-1 OOF probs) | same candidates/folds/dev as E003 | **dev F0.5 0.9547** (+0.0071; train OOF 0.9537); US 0.9622, India 0.9432 | — | 27 min | expected-F decision now best (thr 0.65: 0.9539). Gain mostly from margin vs the record's best competing S1 (62% gain), not sibling similarity |
