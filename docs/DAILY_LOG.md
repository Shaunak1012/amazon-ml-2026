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

**Teammates' extras (27 Sep):** CE03 (e5-large CE) and OW03 (owner model) are trained by a teammate on their GPU. Integrate only if delivered by ~15:00 as per-chunk parquet aligned to runs/E015/{train,test}_chunks (s1_id, cand_id, ce_score; fold 0 + all test; like runs/E016-ce), trained on folds 1-4 only (fold 0 unseen), MIT/Apache <= 8B: add as an extra --ce-dir to the E023 stage-2 command (new --frames dir, ~75 min) and keep only if dev at test-like density improves. SH01 = our E025 (already in). Seed averaging = add in the morning.
**Step 0 (user-approved, 27 Sep): E027 = our own multilingual-e5-large cross-encoder, start the moment E022x is DONE (~06:30).** (a) download intfloat/multilingual-e5-large (MIT, 560M). (b) train: python -m src.er_crossenc train --chunks runs/E015/train_chunks --out runs/E027-ce-large/model --n 1500000 --exclude-folds 0 --model-name intfloat/multilingual-e5-large --batch 32 --lr 1.5e-5 (~3.5 h; check GPU shared-memory spill; lower batch if needed). (c) score ONLY the final candidate rows (stage-1 prob >= 0.2 OR runs/E016-ce score >= 0.01) of fold 0 + test: add a keep-mask option to er_crossenc cmd_score (NaN for other rows, row-aligned like --only-folds) -> runs/E027-ce/{train_ce,test_ce} (~3.5 h). (d) stage 2 = E023a command + extra --ce-dir runs/E027-ce, new --frames dir (~75 min, CPU) -> keep only if dev at test-like density improves; candidate_pairs.tsv must stay byte-identical to E023a's. Deadline to have it: ~15:00; otherwise skip. Conflicts: the optional clean-filter GPU scoring is dropped in favour of this. Move monitor/ into src/ only AFTER E027 finishes (its jobs use python -m monitor.launch). Slots: 1 E023a ~09:30, 2-3 unseen sweep ~10:00, 4 E027 variant ~15:00, 5 final <=22:00.
**27 Sep 03:25:** E023a-core hung after writing its outputs (O(n^2) in the new unseen-variant step; fixed 6bdd28d), killed 03:20; outputs finished from saved probs (runs/E015/sub_E023a_core, both validators PASS, all candidate targets met); E023a-ST relaunched 03:25 with the fix (ETA ~04:00).
**27 Sep 19:10 (final):** sub-16 E033 0.987692, sub-17 E034 (+OW04 owner model) **0.98793** best. E035 (+CE03) flat. E036 France self-training round 2 (teacher E034, 450k French pseudo-labels) + E037 French candidate rescue -> **final candidate E037** (tag sub-18, dev 0.99154, 3.982 cands/S1, deterministic, validators PASS). E038 (round 3, teacher E037) running; ships only if French uncertainty drops further + drift guard vs E037 PASS; else E037. One upload left (the final), by ~21:30.
**27 Sep 14:20 (final path):** sub-12 0.98631, sub-13 (France 0.55) 0.985959, sub-14 (France 0.90) **0.986359** best; teammates use the other slots, ONE final upload left. Organisers confirmed self-training + synthetic pairs allowed; candidate_pairs = set fed to the matching model; smaller ranks higher -> filter stage-1 >= 0.5 OR E016 CE >= 0.05 (3.98 cands/S1 test, RR 99.99996%). **Final = E033** (E030 self-trained francized e5-large CE, dev 0.99099, French uncertain pairs/S1 0.331) in submissions/sub_E033_e030_c383_unseen, unless E031-final (~16:50) beats it on dev (then refit on the 3.83 filter). Then: move monitor/ into src/, tests, tag sub-16, zip --ref sub-16, sha256; user uploads by ~20:00. Commits: check `git branch --show-current` == shaunak, push HEAD:shaunak (Codex branch incident).
**27 Sep 04:00:** E023a-ST done 03:51 (= core dev, identical candidates, validators PASS). E027 pairs pre-sampled, e5-large cached. France variants u80/u90/u95 built + validated. Global-threshold sweep on E023a dev: 0.70 0.99080 / **0.75 0.99086** / 0.80 0.99079 / 0.85 0.99066 / 0.90 0.99032 / 0.95 0.98951 -> 0.75 stays (slot 4 'global threshold' has no value; slot 4 = E027 or a France variant). G0 + E027 chains armed (runs/E023b-chain.log, runs/E027-chain.log).
**Itinerary (revised 27 Sep 01:00, new chat):** E022x train split done 00:56 (54.8 prompts/s); test split 1.04M prompts
-> "E022x CHAIN DONE" ~06:20. E023a-core dev **0.99086** at test-like density (best on the LB-honest basis; E025 0.99039),
cands 4.71/S1 fit, 4.74/S1 test; ETA ~01:15. E023a-ST ~02:30. 06:20 launch G0 (scripts/chains/e023b_full.sh, CPU, ->
~07:40) + E027 (scripts/chains/e027_ce_large.sh, GPU: train ~3.5-3.8 h -> ~10:05; scoring ~3.5 h -> **~13:30**; stage 2
-> **~15:00**). **Re-check E027 training speed after ~1,000 steps and scoring speed after its first shard; adjust ETAs.**
Slots: 1 = sub-12 ~09:30 (better of E023b / E023a-core by dev); 2 ~10:30 and 3 ~12:00 = threshold variants; **4 = E027
~15:00** (if dev improves, else a threshold variant); 5 = final <= 22:00. Stage-2 launch cutoff 19:30.
All targets per result: dev F0.5 (test-like density), LB, cands/S1 4.6-4.8, recall >= 0.996 dev + fit pool
(scripts/cand_recall.py), reduction ratio >= 99.9998%, organiser validator PASS --check-ids, byte-identical candidates.
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
