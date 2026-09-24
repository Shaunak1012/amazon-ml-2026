# Experiment log

Rules: one row per experiment (all folds). ID = `E###-short-name`, config at `configs/E###-short-name.yaml`,
OOF at `OOF_DIR/E###-short-name/`. CV = mean over folds (± std). LB only if submitted. Runtime = wall-clock GPU/CPU.
Commit the row in the same commit as the code/config (`exp: E### <desc> CV <score>`).

| id | date (IST) | owner | model | features / inputs | CV (± std) | LB | runtime | notes |
|---|---|---|---|---|---|---|---|---|
| E000-template | 2026-09-24 | setup | MLP (synthetic) | synthetic | — | — | — | scaffold smoke test only |
