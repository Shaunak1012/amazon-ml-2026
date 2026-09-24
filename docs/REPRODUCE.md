# Business Entity Resolution: how to reproduce

This file becomes `code/business_entity_resolution/README.md` in the submission zip.
**Fill in the exact commands once the final pipeline is fixed.**

## Environment
- Python 3.11; `pip install -r requirements.txt` (pinned). GPU optional (used only for embeddings: <fill in>).
- Tested on: Windows 11, RTX 5080 16 GB, Ryzen 9 9950X3D, 64 GB RAM.

## Data
Put the organisers' dataset at `data/dataset/{train,test}/` (or set `DATA_DIR` in `.env`, see `.env.example`).

## End-to-end: data → blocking → matching → output
```bash
# <fill in: e.g.>
python -m src.<pipeline> --config configs/<final>.yaml --out output/
python -m src.er_submission validate --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir data/dataset/test
```
Expected runtime: <fill in>. Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`.

## Models used (licence, parameters)
| Model | Licence | Params | Role |
|---|---|---|---|
| <fill in> | MIT/Apache-2.0 | ≤ 8B | |

No external data, APIs or lookups are used; everything is derived from the provided training data.
