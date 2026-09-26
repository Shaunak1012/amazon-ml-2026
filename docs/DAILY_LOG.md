# Daily log

Short entries at the end of each block, newest first: **tried / worked / next**. A teammate joining
mid-way should catch up in 2 minutes. Also keep the GPU queue current.

## GPU queue (update whenever it changes)
| now running | ETA (IST) | next up | owner |
|---|---|---|---|
| — | — | — | — |

## HANDOFF — 2026-09-26 22:05 IST (read this first in a new session)
**Leaderboard:** best = sub-10 (E020) **0.985645**; leader 0.990748. Our slots: ~1/day (team shares 5/day).
**Goal (user):** candidate_pairs ~4.71/S1 (recall 0.9963, reduction 99.99988%) and matching as close as possible to the
0.99892 dev ceiling. Report BOTH files for every experiment (dev F0.5, LB, cands/S1, candidate recall, ceiling).

**Kept for the final build (E023):** E020 CE dirs (runs/E015-ce + runs/E020-ce) + E022 reranker feature (Qwen3-4B LoRA,
dev 0.99116, +0.00019) + E024 --norm2 + E025 --comp-keep 0.78 (+0.00064 at test-like density) + comp-cols
(cos_name name_ratio name_jw name_full_tset name_tsort) + candidate filter --prune-eps 0.2 --prune-ce-dir runs/E016-ce
--prune-ce 0.01 (4.71/S1). E026 tuning = no gain (defaults). Stage-2 seed averaging still to add to cmd_stage2.

**Running overnight (chains, watched):** runs/E022b-chain.log (E022 test scoring -> to-ce -> E022-dev), then
runs/E022x-chain.log (reranker on wider band, data runs/E021-llm/data_ext -> merged feature runs/E022-llmce-full, ~07:00).
Batch 32 (batch 64 spilled GPU memory). Keep-awake on; don't close the app.

**Team (for the doc):** Shaunak A. Rai, Shaswat Solanki, Shivam Anand, Shreyas Sreenivas. **Slots 27 Sep: 5.** Portal takes matching_results.tsv only (an early candidate/zip upload errored); final zip upload expected on the last day.
**Reviews 26 Sep night:** packaging (README stale; unrecorded steps E014/E015/E021 data10/data_ext; hand-copied pseudo-label parquet; need scripts/reproduce.sh; pin HF revisions; exclude .ps1), compliance (self-training on test = DQ risk vs 'provided training data' -> recommend drop; stage-2 context features read the 15 not the 4.71 -> prune chunks before features; FAISS scalability benchmark; fix 'learned abbreviation maps' wording).
**Tomorrow (27 Sep):** move monitor/ into src/ after chains end; offline flag; seed averaging; E023 build with --out
-> validators -> sub-11 (~10:00); final refit incl. dev S1s; README/REPRODUCE (full dependency chain incl. E016
pseudo-label source); Documentation_template.md (2-page core + appendices; need team names); zip via
scripts/make_submission_zip.py (dry-run passed); final upload <= 22:00; final zip from the same commit/files.
Rules/compliance: docs/COMPETITION.md (organiser update on candidate size included). Docstrings done (107).

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
