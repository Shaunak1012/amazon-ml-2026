# Business Entity Resolution: how to reproduce

This file becomes `code/business_entity_resolution/README.md` in the submission zip. It reproduces the current
pipeline (submission `sub-02`, experiment E004). Later submissions update this file at their own git tag.

## Environment
- Python 3.11. `pip install -r requirements.txt` (pinned; includes PyTorch 2.8.0 + CUDA 12.8 wheels).
- **GPU required in practice (CUDA)**: multilingual-e5-small embeddings (~24M texts) and exact nearest-neighbour
  search. Tested on an RTX 5080 (16 GB). Without CUDA the code falls back to CPU, but embedding becomes impractically slow.
  Everything else runs on CPU (32 threads used by LightGBM and rapidfuzz).
- RAM: 64 GB (peak ~50 GB during test prediction). Disk: ~40 GB free for caches.
- Copy `.env.example` to `.env`. `DATA_DIR` defaults to `./data`.

## Data
Place the organisers' files at `data/dataset/train/*.tsv` and `data/dataset/test/*.tsv`.

## End-to-end: data → blocking → matching → output
```bash
python -m src.er_data                                   # 1. exact TSV load + parquet cache, row counts verified (~15 s)
python -m src.er_normalize                              # 2. transliterate + normalise names/addresses (~2 min, 30 processes)
python scripts/make_folds.py                            # 3. 5 folds over train S1 (seed 42) + 100k dev subset
for v in name addr both; do                             # 4. multilingual-e5-small embeddings, 3 views, train + test
  python -m src.er_embed --split train --sources 1 2 3 --view $v --model small
  python -m src.er_embed --split test  --sources 1 2 3 --view $v --model small
done                                                    #    (~75 min total on the RTX 5080)
python -m src.er_pipeline predict --exp E004-predict-sub02 --k 10 --train-s1 300000 \
    --out output/                                       # 5. blocking + features + LightGBM + decisions (~70 min)
python -m src.er_submission validate --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir data/dataset/test
```
Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

Validation experiment (held-out dev score, same recipe):
`python -m src.er_pipeline validate --exp E003-baseline --k 10 --train-s1 200000` gives dev F0.5 0.9476.

## Pipeline summary
1. **Normalisation** (`src/er_normalize.py`): anyascii transliteration of all scripts, lowercasing, punctuation and
   `null` removal, legal-suffix stripping for a "core name", number and postcode extraction.
2. **Blocking** (`src/er_blocking.py`, `src/er_embed.py`): multilingual-e5-small embeddings of three views (name,
   address, name + address). Exact top-10 cosine search per view, per source (S2, S3), within the same country
   string. The union gives about 50 candidates per S1 and 97.4% pair recall on the dev split.
3. **Matching** (`src/er_features.py`, `src/er_model.py`): 31 features (embedding cosines per view, rapidfuzz name
   and address similarities, number and postcode agreement, within-S1 and within-record rank context). LightGBM,
   5 fold models with S1-grouped folds.
4. **Decisions** (`src/er_decide.py`): probability threshold 0.70, tuned on out-of-fold predictions for macro F0.5,
   with each S2/S3 record assigned to at most one S1.

## Models used (licence, parameters)
| Model | Licence | Params | Role |
|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 118M | text embeddings for blocking and similarity features |
| LightGBM (library) | MIT | n/a (gradient-boosted trees) | pair classifier |

No external data, APIs or lookups are used; everything is derived from the provided training data.
