# Daily log

Short entries at the end of each block, newest first: **tried / worked / next**. A teammate joining
mid-way should catch up in 2 minutes. Also keep the GPU queue current.

## GPU queue (update whenever it changes)
| now running | ETA (IST) | next up | owner |
|---|---|---|---|
| — | — | — | — |

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
- **Next (Day 2):** France probe on sub-06; error analysis of E010 dev; multi-key/char-ngram blocking for the 2.6%
  never retrieved; stronger cross-encoder (2-fold CE on all training folds, more pairs); final refit on all folds.

## 2026-09-24 (pre-launch)
- **Did:** built scaffold — metrics/CV/OOF/submission validator with tests, heartbeat + watchdog + Discord alerts, playbooks.
- **Worked:** tests green; monitor demo fires alerts for clean run, OOM crash, NaN, hang, hard kill; live Discord
  alerts verified. Image downloader tested locally. 6 backbones (3.6 GB) pre-cached and verified to load offline on
  GPU in bf16: e5-base-v2, bge-base-en-v1.5, MiniLM-L6, deberta-v3-base, siglip-base-224, clip-vit-large-14.
- **Next (Day 1, 00:00 IST):** follow the Day-1 checklist in CLAUDE.md.
