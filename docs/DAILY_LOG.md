# Daily log

Short entries at the end of each block, newest first: **tried / worked / next**. A teammate joining
mid-way should catch up in 2 minutes. Also keep the GPU queue current.

## GPU queue (update whenever it changes)
| now running | ETA (IST) | next up | owner |
|---|---|---|---|
| — | — | — | — |

## HANDOFF — 2026-09-26 23:40 IST (read this first in a new session; supersedes all earlier notes)
**LB:** sub-10 (E020, self-trained) **0.985645** = best; sub-11 (E017, same w/o self-training) 0.985144 -> self-training
= +0.0005 LB. Leader 0.990748; **user target 0.9908 (rank 1)**. Slots 27 Sep: **5**. Deadline 27 Sep 23:59 IST.
**Goal (both files equal priority):** candidate_pairs ~4.71/S1 (stage-1>=0.2 OR E016 CE>=0.01; dev recall 0.9963, fit-pool
0.9962; test FR 5.73/IN 4.60/US 4.53) + matching toward the 0.99892 dev ceiling. Report dev F0.5, LB, cands/S1, recall.
**Decided (user):** drop self-training UNLESS organisers confirm it is allowed (Google Form); clean pre-feature filter =
optional; both TSVs from ONE stage-2 run; zip from one commit/tag; no stage-2 launch after 19:30.

**Running overnight (chains in runs/*-chain.log, launched from the OLD chat; keep it open):**
1. E022b: DONE 23:40 (E022 dev F0.5 0.99116 with full test mapping; confirms the early check).
2. E022x (GPU): reranker on the wider band (data runs/E021-llm/data_ext) -> runs/E022-llmce-full (ETA ~06:30; started ~23:25).
3. E023a-core (CPU, launched 23:41, ETA ~01:00): stage2 --ce-dir runs/E015-ce runs/E016-ce runs/E022-llmce --norm2
   --comp-keep 0.78 --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --prune-eps 0.2 --prune-ce-dir
   runs/E016-ce --prune-ce 0.01 --unseen-threshold 0.85 -> submissions/sub_E023a_core (+ _unseen), validators in
   runs/E015/sub_E023a_core/validate_*.txt, probs/stage2.json copied there.
4. E023a-ST (CPU, after 3, ETA ~02:30): same with runs/E020-ce (self-trained) -> submissions/sub_E023a_st; candidate file
   must be byte-identical to E023a-core (chain prints it).
Watchdogs: GPU-spill + health (background), Discord via monitor.notify. Keep-awake on.

**Morning steps (in order):** (1) check chain logs + validators; record E023a dev (test-like density) + cands/S1.
(2) G0: dev of runs/E022-llmce-full vs core; if full >= core, rebuild E023a with it (~75 min, new --frames dir).
(3) Slot 1 = E023a (ST or not per organisers), tag sub-12, safety zip from the tag (scripts/make_submission_zip.py
--ref sub-12 --team SHSHSHSH --outputs <dir> --doc docs/Documentation_template.md --test-dir data/dataset/test).
(4) Slots 2-3: unseen-country threshold sweep via scripts/decision_variants.py (same probs => identical candidates),
e.g. --t-unseen 0.85 / 0.90; slot 4: global threshold picked on test-like-density dev; slot 5 final <= 22:00.
(5) After all chains end: move monitor/ into src/ (update imports + chains), pin/offline notes; fill 14 {{FINAL:..}}
placeholders in docs/Documentation_template.md + docs/REPRODUCE.md (drafts committed; scripts/reproduce.sh drafted);
FAISS scalability note; zip from final tag; sha256 of TSVs = uploaded file. Team: Shaunak A. Rai, Shaswat Solanki,
Shivam Anand, Shreyas Sreenivas.
Regression check 26 Sep: today's code reproduces E020 dev exactly (new flags are opt-in).

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
