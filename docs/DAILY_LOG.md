# Daily log

Short entries at the end of each block, newest first: **tried / worked / next**. A teammate joining
mid-way should catch up in 2 minutes. Also keep the GPU queue current.

## GPU queue (update whenever it changes)
| now running | ETA (IST) | next up | owner |
|---|---|---|---|
| — | — | — | — |

## HANDOFF — 2026-09-26 12:10 IST (read this first in a new session)
**Leaderboard:** best = sub-08 **0.983** (E015). #1 = 0.988419. sub-07 0.975726. Slots on 26 Sep: **3 left** (assumes
sub-07 counted on 26 Sep). Keep >= 2 for 27 Sep.

**E015 (sub-08):** fine-tuned bi-encoder view `both_ft` (E014) added to retrieval + features; stage 1 AND stage 2 fit on
**fold 0 minus dev** (341k S1 unseen by the bi-encoder and the CE) because the ft cosine is inflated on folds 1-4.
Dev F0.5 **0.9898** (OOF 0.9896; US 0.9892, India 0.9907). Stage 3 (anchors from stage-2 probs) = no gain.
Artefacts: `runs/E015/` (chunks, models), E015's own stage-2 outputs backed up in `runs/E015/sub08_E015/`.

**Running now:** E016 = cross-encoder v2 (multilingual-e5-base, 3M pairs, folds 1-4, E015 hard negatives;
`runs/E016-ce-base/model`). Chain `scripts/chains/e016_score.sh` scores fold 0 + test -> `runs/E016-ce/`, then
`scripts/chains/e016_stage2.sh` runs stage 2 with BOTH CEs (`--ce-dir runs/E015-ce runs/E016-ce`) -> `submissions/sub_E016`
(~13:30). It writes into runs/E015 (stage2.json etc.). Check: `cat runs/E016-score-chain.log runs/E016-stage2-chain.log`.
Submit E016 only if dev beats 0.9898 by more than noise (~0.0003).

**Ops lessons:** PC slept 09:21-12:00 and froze jobs -> `scripts/keep_awake.ps1` while jobs run (and set Sleep=Never).
RAM is the bottleneck (64 GB): stage 2 peaks ~48 GB; `scripts/ram_guard.ps1` suspends a lower-priority job when low;
`scripts/pause_jobs.ps1 pause|resume|status` pauses everything. Stage 1 now reuses saved fold models on restart.

**Next ideas (ranked):** 1) E016 result; 2) France: pseudo-label fine-tune on confident French test pairs (rules say
models "fine-tuned only on the provided data" -> test inputs are provided data; confirm wording first); 3) second
bi-encoder view / k=20 for the last recall; 4) final approach doc + zip (`scripts/make_submission_zip.py`).

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
