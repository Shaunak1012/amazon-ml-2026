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
