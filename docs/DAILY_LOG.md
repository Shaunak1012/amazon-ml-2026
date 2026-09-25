# Daily log

Short entries at the end of each block, newest first: **tried / worked / next**. A teammate joining
mid-way should catch up in 2 minutes. Also keep the GPU queue current.

## GPU queue (update whenever it changes)
| now running | ETA (IST) | next up | owner |
|---|---|---|---|
| — | — | — | — |

## HANDOFF — 2026-09-25 22:15 IST (read this first in a new session)
**Leaderboard:** best = sub-06 **0.975154** (rank 53; #1 = 0.986955; top-50 = PPI cut). Probes: sub-03 0.947598,
France-emptied 0.821 (France ≈ 0.89), France-strict 0.949822. Slots: 0 left on 25 Sep; **5 fresh on 26 Sep**.

**sub-07 uploaded 26 Sep 00:0x → LB 0.975726 (new best, +0.00057 vs sub-06); 4 slots left on 26 Sep.** Was: `submissions/sub_E013_rarity/matching_results.tsv` = tag **sub-07**
(E013: sub-06 + name-rarity features, dev F0.5 0.9844; both validators pass). Expected LB ~0.9755.

**Running now (from the old chat session; keep that window open until it finishes):**
`E014` fine-tuned bi-encoder chain → `runs/E014-bienc/model` → train embeddings `data/cache/emb/train_s*_both_ft.npy`
→ prints dev recall gain (`scripts/recall_gain.py --view both --tag ft --k 10 20`). Check with:
`ls runs/E014-*; cat runs/E014-bienc-train/exit.json; tail runs/E014-bienc-embed-train/train.log`.

**E014 RESULT (22:42): ft view dev recall alone top-10 = 0.9959; union 0.9740 -> 0.9976 (recovers 90.7% of misses). DO THE REBUILD.** Test embedding with the ft model was started from the old session (check `runs/E014-bienc-embed-test/exit.json`; files `data/cache/emb/test_s*_both_ft.npy`). Rebuild steps:
1. embed TEST with the ft model (`python -m src.er_embed --split test --sources 1 2 3 --view both --model runs/E014-bienc/model --tag ft`);
2. stage 1 over the full population with the ft view added (needs a `--views` option in `src/er_fullpass.py` stage1 and
   `Split(views=...)`; ft files use suffix `ft`, so Split must map view→suffix, e.g. views name/addr/both(small)+both(ft));
3. CE A scores for the new chunks (train folds 0,3 and test), then `stage2 --ce-dir ... --fit-folds 3` (+ rarity is built in).
If the gain is small, skip the rebuild.

**Error analysis (E010 dev, loss 0.016):** 51% true pairs never retrieved (India 3x US), 37% retrieved but rejected,
10% false matches. Empty-address candidates are a big share of rejects/FPs (→ name-rarity features, E013).
Char n-gram view (E012) only +0.4pt recall — not worth a rebuild alone.

**Key facts:** pipeline & commands in docs/REPRODUCE.md; experiments in docs/EXPERIMENTS.md; ledger in
docs/SUBMISSIONS.md; rules/compliance in docs/COMPETITION.md. Cross-S1 features must be computed over the full
population (DECISIONS 2026-09-25). Portal takes the TSV only. Git: sequential commits, user as sole author, no AI mentions.

## 2026-09-25 (Day 1) — end of day
- **Leaderboard:** sub-06 **0.975154** (leader 0.986955). sub-03 0.947598; France probes: France-emptied 0.821,
  France-strict 0.949822. sub-02 failed at the portal (counted). 0 slots left today; 5 tomorrow.
- **Pipeline now (sub-06):** normalise → multilingual-e5-small 3-view exact GPU blocking (97.4% pair recall) →
  stage-1 LightGBM over ALL S1 (E007, out-of-fold) → top-15 → cross-encoder score (E008, e5-small fine-tuned on
  folds 1-2) → stage-2 LightGBM with cluster + full-population competition features (fit on fold 3) → expected-F0.5
  decisions with one-S1-per-record. Dev F0.5 0.9840.
- **Learned:** (1) cross-S1 features must be computed over the full population (sampling bias fixed in E007).
  (2) France (unseen) was the weak spot (~0.89) due to false merges; the cross-encoder fixes most of it.
  (3) dev→LB gap on US+India ~ -0.005.
- **Evening:** E011 (two fold-disjoint cross-encoders + stage 2 on all folds) = dev 0.9841, **no gain** over E010 (0.9840):
  the matcher is saturating. Rank 53 (top-50 = PPI cut).
- **Next (Day 2):** save stage-2 dev predictions → error analysis → recall work (multi-key + char n-gram blocking;
  oracle ceiling with current candidates 0.9915) → France probe on the best model.
- (earlier plan) France probe on sub-06; error analysis of E010 dev; multi-key/char-ngram blocking for the 2.6%
  never retrieved; stronger cross-encoder (2-fold CE on all training folds, more pairs); final refit on all folds.

## 2026-09-24 (pre-launch)
- **Did:** built scaffold — metrics/CV/OOF/submission validator with tests, heartbeat + watchdog + Discord alerts, playbooks.
- **Worked:** tests green; monitor demo fires alerts for clean run, OOM crash, NaN, hang, hard kill; live Discord
  alerts verified. Image downloader tested locally. 6 backbones (3.6 GB) pre-cached and verified to load offline on
  GPU in bf16: e5-base-v2, bge-base-en-v1.5, MiniLM-L6, deberta-v3-base, siglip-base-224, clip-vit-large-14.
- **Next (Day 1, 00:00 IST):** follow the Day-1 checklist in CLAUDE.md.
