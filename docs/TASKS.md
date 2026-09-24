# Task board

Owner: `local` or `cloud`. Status: `todo` → `cloud` / `wip` → `review` → `done`. Protocol: docs/WORKERS.md.
One row per task; the local worker keeps this current (cloud workers never edit it).

| id | task | owner | branch | files | status | notes |
|---|---|---|---|---|---|---|
| T001 | EDA + CV design on real data | local | main | notebooks/, docs/DECISIONS.md | todo | needs student_resource |
| T002 | Normalisation functions + tests | — | — | src/er_normalize.py | todo | good cloud candidate once EDA shows real noise examples |
| T003 | Blocking retrievers + recall curve | local | main | src/er_blocking.py | todo | K chosen on real data |
| T004 | Feature groups (rapidfuzz, TF-IDF, numbers) | — | — | src/er_features.py | todo | good cloud candidate |
| T005 | LightGBM matcher + OOF | local | main | src/er_model.py | todo | |
| T006 | F0.5 decision layer | local | main | src/er_decide.py | todo | threshold tuned on OOF |
| T007 | Pipeline CLI + outputs | — | — | src/er_pipeline.py | todo | |
