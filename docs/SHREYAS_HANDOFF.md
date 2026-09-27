# Shreyas branch — full handoff (Amazon ML Challenge 2026)

Written 27 Sep 2026 ~16:00 IST. Covers the whole working session (26 Sep ~14:00 → 27 Sep ~16:00 IST).
Read this first in a new session, then `docs/SHREYAS_LOG.md` (per-experiment detail, newest first) and `CLAUDE.md`.
**No secrets in this file.** Presigned links live only in local, uncommitted files (§10). The user's VPN/SSH password
was shared in chat once; it was **never used, never stored**, and must not be written anywhere.

---

## 0. TL;DR — state at 16:00 IST, 27 Sep

- **Deadline:** portal closes **27 Sep 23:59 IST**. Team plan: final upload by ~20:00, decision by ~19:30.
- **Team best LB:** sub-14 **0.986359** (Shaunak, E023b-full + France threshold 0.90). Leader ~0.9915. >10 teams > 0.99.
- **Team final candidate:** Shaunak's **E033** (dev 0.99099, test 3.98 cands/S1, filter "3.83": stage-1 ≥ 0.5 OR E016 CE ≥ 0.05).
- **Pending, decisive:** Shaunak's **E034** = E033 + our **OW04** owner-model features (`--ce-dir ... runs/OW04p runs/OW04m`).
  Chain `scripts/chains/e034_ow04.sh` on origin/shaunak (pushed 15:50). It waits for his E031-final (~16:50), then a full
  frame rebuild + stage 2 → **result expected ~17:45–18:30**. Built-in gate (all must hold, else ship E033):
  dev ≥ E033 + 0.0002 (≥ 0.99119), OOF ≥ E033, US and India each ≥ E033 − 0.0001, organiser validator PASS,
  candidate_pairs.tsv byte-identical to E033's (3.83 filter preserved).
- **Also pending on his side:** E031-final (~16:50): per his handoff, E033 stays the final "unless E031-final beats it on
  dev (then refit on the 3.83 filter)". E034 starts only after E031 finishes (RAM).
- **Our own finals (built, validated, NOT uploaded):** v1 and v2 in `s3://<bucket>/shreyas/artifacts/final_shreyas{,_v2}/`
  (§7). Recommendation given: don't upload them standalone; E033 or E034 is better (his pipeline has France threshold,
  norm2, reranker, self-training, smaller candidate set).
- **Submission slots:** the user reported **2 left** (Shaunak's log says "one upload left" for him). Plan: slot 1 = E034
  if its gate passes, else E033; slot 2 = reserve.
- **Honest ceiling:** Shaunak's dev-loss analysis (A031) shows ambiguous empty-address records cap dev at ~0.9911, so
  the user's requested "validation ≥ 0.993" is not reachable; realistic LB for E034 ≈ 0.9870–0.9876.
- **Machines:** SageMaker r7i box (idle, runner up) and college GB10 GPU (idle, runner up in tmux). Nothing running.

---

## 1. Goal, rules, constraints

- Task: business entity resolution. For each Source 1 (S1) record, list matching S2/S3 records. Metric: macro F0.5
  per S1 (singletons score 1 only if predicted empty). Test adds France (unseen in train).
- Deliverables: `matching_results.tsv` (scored) + `candidate_pairs.tsv` (exact set the final model scores) + zip
  (code/README/requirements/methodology doc).
- **Organiser update (26 Sep):** candidate generation counts in final ranking — smaller candidate set per S1 ranks
  higher; candidate_pairs.tsv and the code producing it are reviewed. "If several filtering stages, candidate_pairs.tsv
  is the last one."
- **Rules:** MIT/Apache models ≤ 8B params; no external data/lookups/geocoding; country is an open string (no hard-coding).
- **Self-training on test pseudo-labels:** team first dropped it (26 Sep 22:40), then organisers **confirmed it allowed**
  (Shaunak's log, 27 Sep). Our finals do NOT use self-trained columns (we dropped `ce_score_2` of the E020 frame).
- 5 submissions/day, shared by the team.

## 2. User instructions and preferences (must keep following)

- **Only change the `shreyas` branch.** Read every branch; never merge `origin/shaunak` wholesale (read with
  `git show origin/shaunak:<file>`, copy code into our own files). One earlier merge (6349fbf, 26 Sep) exists on shreyas.
  Enforced by `scripts/githooks/pre-push` (`git config core.hooksPath scripts/githooks`).
- Team CLAUDE.md rule: **no `Co-Authored-By` / AI mentions** in commits; conventional commit messages.
- Model routing: Opus for reasoning/code, cheapest model (Haiku) for monitoring.
- Use AWS credits/compute freely (~$180 at start). User wanted everything GPU-heavy on the college GB10 box.
- **Use tmux for persistent processes on the GPU box** (the lab box kills an SSH session's processes at logout).
- **Alert the user if the GPU box goes silent** (heartbeat > 5 min old) so they can log in (VPN session lasts ~4 min,
  then a ~5 min lockout).
- Targets the user set: LB > 0.993 (not reachable — say so honestly), candidate lists ≤ 4.7/S1 (team uses 3.83 filter),
  "validation ≥ 0.993 before uploading" (explained it blocks all uploads; dev ceiling ~0.9911).
- Rules compliance is paramount; honesty about numbers (no overclaiming).

## 3. Repos, branches, local paths

- Team repo: https://github.com/Shaunak1012/amazon-ml-2026 (public). Local clone: `C:\Xhreyas\projects\team-repo`,
  branch `shreyas`. Latest shreyas commit at writing: `8eb295e` (+ this handoff commit).
- Teammates' branches: `shaunak` (main pipeline, very active), `main` (PR-only), `solanki` (one LFS data commit), `shivam`.
- User's personal repo: `C:\Xhreyas\projects\amazon` (origin ShreyasSreenivas23/AmazonML) — holds organiser PDFs,
  `student_resource/dataset/` (raw TSVs), a pointer `CLAUDE.md`, and the private link files (§10). Never push it.
- Local venv for tests: `C:\Xhreyas\projects\team-repo\.venv` (Windows; CPU torch + transformers installed).
  `python -m pytest -q tests/test_er_stage2.py tests/test_er_owner.py tests/test_er_decide2.py` → all pass.
- Scratchpad (session-only): `C:\Users\shrey\AppData\Local\Temp\claude\C--Xhreyas-projects-amazon\...\scratchpad`
  (had presigned URL files; do not rely on it).

## 4. Infrastructure

### 4.1 AWS
- Account 548171706001, region **ap-southeast-2**. Local AWS CLI identity: IAM user **`shreyas`** (S3 only; no SageMaker,
  IAM, EC2, ServiceQuotas APIs). User later added inline policy `ReadShaunakShared` to user `shreyas` and role
  `AmazonSageMakerAdminIAMExecutionRole` (read his bucket's shared/*), but cross-account reads still failed; unused.
- Bucket: **`amazon-sagemaker-548171706001-ap-southeast-2-bjmz86wfr8l6oy`** (SageMaker Unified Studio project bucket,
  BucketOwnerEnforced, SSE-S3). Bucket policy allows Shaunak's account (908861959318) PutObject to `shared/*` only.
- SageMaker Unified Studio JupyterLab space on **ml.r7i.8xlarge** (32 vCPU, 256 GB RAM, no GPU, ~$2–2.6/h). Switching
  the space to ml.g5.2xlarge failed (InsufficientCapacity in all ap-southeast-2 AZs). SageMaker **training-job quota for
  ml.g5.2xlarge is 0** (spot and on-demand; probe jobs failed with ResourceLimitExceeded).
- **Cost:** the r7i space has been running since ~26 Sep 15:00 IST (~25 h ≈ $55–65 so far). **Stop the space after the
  deadline** (and delete large S3 prefixes later if credits matter).

### 4.2 r7i S3 job runner (`scripts/sm/`)
- `bootstrap.sh` (in S3 at `shreyas/bootstrap.sh`): in the space terminal run
  `aws s3 cp s3://<bucket>/shreyas/bootstrap.sh - | bash` → repo code snapshot from S3 into `~/amazon-ml-2026`,
  `.venv` (requirements-core + CPU torch + transformers 5.17), data unzipped to `data/dataset`, starts
  `scripts/sm/runner.py` (nohup). Re-run after any space restart (runner dies with the space).
- Protocol under `s3://<bucket>/shreyas/`: `queue/<job>.sh` (picked up within 20 s; runs `bash` in repo root; several in
  parallel), `logs/<job>.log` (synced every 60 s), `done/<job>.json` (exit code), `heartbeat.json` (60 s),
  `artifacts/...` (job outputs), `control/stop` (stops runner). Box clock is UTC (IST − 5:30).
- Code delivery: `bash scripts/sm/push_code.sh` (local) publishes `git archive HEAD` to `shreyas/code/repo.tar.gz`;
  jobs call `bash scripts/sm/pull_code.sh`. **Commit before push_code** (it archives HEAD).
- Queue a job: `aws s3 cp scripts/sm/jobs/<x>.sh s3://<bucket>/shreyas/queue/<NN>_<x>.sh`.
- Box state: `runs/import/export/{frames -> runs/shared/frames/E019, train_min.parquet}`, `runs/shared/...` (Shaunak's
  uploads), `runs/frames2/<tag>/` (every stage-2 run: result.json, dev_probs.parquet, models), `runs/OW0x`, `runs/CE03dev`,
  `submissions/final_shreyas{,_v2}`, `data/cache` (parquet + norm caches, folds = Shaunak's exact file).

### 4.3 College GPU box (NVIDIA GB10, 128 GB unified, 20 ARM cores, Ubuntu 24.04, ~97.5 TFLOPS bf16)
- Reached by the user via Sophos VPN + SSH as `student-12` (host 10.1.72.54); work dir `~/Desktop/aml`.
  **Claude never logs in**; it controls the box only through a credential-free S3 runner.
- `scripts/gpu/remote_runner.py`: polls a **presigned GET** of `shreyas-gpu/queue/current.sh` every 20 s (each new content
  runs once, by sha256); uploads logs/done/heartbeat through a **presigned POST** restricted to `shreyas-gpu/out/`.
  No AWS keys on the box. Links expire **~3–4 Oct**.
- Bootstrap: user runs (in `~/Desktop/aml`) the command in `C:\Xhreyas\projects\amazon\GPU_BOOTSTRAP_COMMAND.txt`
  (detached, idempotent; torch 2.14 cu130 for GB10 in `repo/.venv`; reports to `shreyas-gpu/out/bootstrap.log`).
- **Runner now lives in tmux session `gpurun`** (moved by `scripts/gpu/jobs/to_tmux.sh`) because the lab box kills
  processes of an SSH session at logout (first runner + first GPU job died that way). If the box reboots, the user must
  log in and re-run the bootstrap command, then queue `to_tmux.sh` (or start the runner in tmux by hand).
- Queue a GPU job (local): `python scripts/gpu/links.py job <script>` (renders `{{GET:<key>}}` and
  `{{GETDIR:<prefix>:<localdir>}}` into 7-day presigned links, uploads to `shreyas-gpu/queue/current.sh`). The runner runs
  each NEW content once — add a nonce comment to re-run the same script. Re-rendering replaces the queue file.
- Outputs: `s3://<bucket>/shreyas-gpu/out/artifacts/{OW03,OW04,CE03}/...`; `scripts/gpu/put.py` splits >1 GB files.
- Known quirk: stdout of some GPU jobs didn't reach the job log; use a status job (`scripts/gpu/jobs/status.sh`) to
  `ls` outputs/processes.

### 4.4 Getting Shaunak's artefacts
- His bucket `amazonml-2026-shared-908861959318` was never readable by us (cross-account denied even after IAM changes).
- Working route: **presigned POST uploader** `scripts/upload_to_shreyas.py` (he runs `--post runs\post_shreyas.json <paths>`,
  keys land under `shared/runs/...`; files > 1 GB split into `.partNNN` + `.done`). Already received: E019 train frame +
  frames.json, folds file, `runs/E015/train_chunks` (12), dev_stage2_E019/E017, sub_E016 test probs, **E020 test frame**
  (3.66 GB) + frames.json + sub_E020 JSONs. (E020 test probs no longer exist on his box.)
- Old export path (`scripts/export_for_shreyas.py`, PUT links) was never used.

## 5. Team (Shaunak) pipeline and latest state (read-only knowledge)

- Retrieval: multilingual-e5-small, 4 views (name/addr/both + fine-tuned bi-encoder E014 `both_ft`), exact top-10 per
  source, same country; union recall 0.998. Stage 1 LightGBM → top-15/S1. CEs: E008 e5-small, E016 e5-base,
  E027 e5-large (1.5M pairs) → E029 francized → E030 self-trained. E022 Qwen3-4B LoRA reranker (+0.00019 dev).
  Stage 2 LightGBM (cluster/sibling + full-population competition + name rarity + norm2) fit on fold 0 minus dev (300k S1),
  dev = 100k fold-0 S1. Decision: expected-F0.5 / threshold, one S1 per record; `--unseen-threshold 0.85` (France).
- Key experiments (his docs/EXPERIMENTS.md): E015 0.9898 (LB 0.983322), E016 0.9905 (LB 0.984727), E017 0.9910,
  E020 self-train (LB 0.985645), E024 norm2, E025 comp-keep 0.78 (+0.00064 on test-like dev), E026 tuning saturated,
  E023a-core 0.99086, **E023b-full 0.99100** (LB **0.98631**, sub-12), sub-13 France 0.55 → 0.985959, **sub-14 France 0.90 →
  0.986359 (best)**, E027 large CE +0.00004, E032 3.98/S1 0.99095, **E033 dev 0.99099, 3.98 cands/S1 test (US 3.76,
  India 3.86, France 4.92), probable final**, A031 dev ceiling ~0.9911. He also knows about our OW/CE03 ("teammates'
  extras"), originally wanted them by ~15:00 in per-chunk --ce-dir format.
- His regime details copied by us: `density_mask` (all fit+dev S1s + random 0.78 of others as competitors, seed 11,
  draws over `allp.s1_id.unique()` in chunk order; only competition features recomputed), `prune_rows` applied AFTER
  features (E023b: stage-1 ≥ 0.2 OR E016 ≥ 0.01 → 4.743/S1; E033: ≥ 0.5 OR E016 ≥ 0.05 → 3.83 filter / 3.98 test).

## 6. Our experiments and results (all dev F0.5; details in SHREYAS_LOG.md)

| ID | What | Result | Verdict |
|---|---|---|---|
| SH01 | test-density stage-2 training (drop 19% of train S1s; pool/S1 train 4.68 vs test ~5.8, ~2.3 vs 1.2 distractors/S1) | test-like dev: current 0.98957 → SH01 0.98986; 4-model ens 0.98999 (+0.00042); paired gain positive 15/15 (3 subsets × 5 rules) | kept; Shaunak built the same idea as E025 |
| SH02 | exact expected-F0.5 (Poisson-binomial) + isotonic vs MC256 | 0.99081 / 0.99082 / 0.99079 | dropped (decision layer saturated) |
| JD01 | release/joint assignment decoding sizing on E016 test probs | only 5,913 records (0.33% S1) lost with a 2nd S1 p ≥ 0.5; 64 at ≥ 0.7; 0 at ≥ 0.9 | dropped |
| CS01/02 | candidate size sweep (stage-1 floor vs cap), E019 frame | top-15 0.99097; floor 0.005 → 5.93/S1 −0.00014; 0.01 → 5.25/S1 −0.00024; caps < 8 break | floor ≫ cap |
| CS03 | test cand/S1 by country per floor | France keeps MORE candidates (0.01: FR 6.81, US 5.55, IN 5.60) | floor safe for France |
| CS04 | strict OR rule (p OR E016 CE; features recomputed on pruned set) | p≥0.02∨CE≥0.02 → 5.08/S1, −0.00008 | strict costs ~nothing |
| CF01 | clean filter (p OR E008 CE, no self-trained model) | 0.02/0.02 → 6.25/S1 dev 0.99096; 0.05/0.05 → 5.43, 0.99082; 0.1/0.1 → 4.70 dev / 4.89 test | used as superset |
| OW00 | r7i CPU benchmark (e5-small) | train ~30 seq/s (L128), infer ~1000 seq/s bf16/AMX | — |
| OW01 | listwise owner model, e5-small, CPU, 48k groups (too-strict leakage rule) | 0.99094 vs 0.99097 (top-15); 0.99100 vs 0.99089 (pruned) | no gain |
| OW02 | CPU owner model, fixed rule (137k groups) | killed (superseded by GPU OW03) | — |
| OW03 | owner model, e5-base, GPU, 2 epochs, 137k groups (17 min) | clean filter 0.99096 → 0.99122 (+0.00026) | kept |
| seeds | 3-seed stage-2 average on OW03 | 0.99111–0.99119 | no gain |
| CE03 | multilingual-e5-large cross-encoder, GPU, 1.2M pairs folds 1-4; scored 2.5M fold-0 + 11.07M test pairs (~55 min/1.23M) | alone +0.00006; with OW03 0.99134 | small; redundant with Shaunak's E027/E030 |
| CF02 | CE03 as candidate filter (p ≥ 0.05 ∨ ce3 ≥ 0.10) | 4.13/S1 dev at 0.99131 (−0.00003), recall 0.9908 | used in our finals |
| OW04 | owner model, e5-large, GPU, 2 epochs (~70 min) | see E23 table / v2 | slightly > OW03 |
| E23 paired | Shaunak's E023b regime copied (comp-keep 0.78, post-feature filter p≥0.2∨E016≥0.01; identical 100k dev rows, 4.706/S1, recall 0.99628) | base 0.99024; +OW03 0.99107 (+0.00083); +CE03 0.99057 (+0.00033); +OW03+CE03 0.99126 (+0.00102); **+OW04 0.99115 (+0.00091)**; **+OW04+CE03 0.99137 (+0.00113)** | owner signal ~3× stronger at test density; base lacks his norm2 + reranker (explains < 0.99100) |
| FINAL v1 | E019 train + E020 test frame minus self-trained ce_score_2; OW03+CE03; CE03 cascade filter; density training | normal dev 0.99117; test-like 0.99094 (density +0.00034 vs 0.99060) | built, validated |
| FINAL v2 | v1 with OW04 | 0.99125 normal (+0.00008), 0.99098 test-like (+0.00004) | within noise of v1 |

Probes: `sub_SH01` (E020 frame + SH01 4-model stage 2; self-trained features) and `probe_france_sh01` (France emptied)
were built + validated for a France LB measurement but **not uploaded** (user wanted validation ≥ 0.993 first).

## 7. Deliverables / artefacts (all under `s3://<bucket>/`)

- `shreyas/artifacts/final_shreyas/` — **v1** `matching_results.tsv`, `candidate_pairs.tsv`, `result.json`. Test: 4.54
  cand/S1 (US 4.33, India 4.37, France 5.56; 1.69% S1 with none), 3.37 matches/S1, 5.76% empty; organiser validator PASS.
- `shreyas/artifacts/final_shreyas_v2/` — **v2** (OW04), validator PASS.
- `shreyas/artifacts/sub_SH01/`, `probe_france_sh01/` — probes (not for final: self-trained features).
- `shreyas/artifacts/handoff/` — for Shaunak: `handoff_OW04p.tar`, `handoff_OW04m.tar`, `handoff_CE03.tar` (~600 MB each;
  per-chunk parquets row-aligned with his `runs/E015/{train,test}_chunks`, cols s1_id/cand_id/ce_score, NaN where unscored;
  coverage OW04 train 2.97% / test 4.02%, CE03 train 7.54% / test 42.59%), plus keyed files `ce3_*.parquet`,
  `ow4_*.parquet`. Test chunks were rebuilt from his E020 frame (chunk k = test S1 positions [200k·k, 200k·(k+1)),
  3,000,000 rows/chunk, last 1,988,160). His `scripts/check_ce_alignment.py` verifies alignment before E034.
- `shreyas-gpu/out/artifacts/{OW03,OW04,CE03}/` — raw GPU outputs (owner features, CE scores, train.json).
- `shreyas/artifacts/*.tsv|json|txt` — every sweep/comparison table (cand_sweep*, clean_filter, ce03_filter, final_dev,
  e23_paired, final_v2_compare, sh01_rules*, jd01_sizing, owner_bench, ow03_seeds, cand_test_by_country).
- Prompt given to Shaunak's Claude: `C:\Xhreyas\projects\amazon\PROMPT_FOR_SHAUNAKS_CLAUDE.md` (steps + acceptance gate +
  fast path via cached frames with a tested `add_extra_ce` snippet). His Claude chose the full-rebuild route.

## 8. Code added on `shreyas` (all tested where marked)

- `src/er_frames2.py` — stage 2 from cached frames on CPU: `--drop-s1-frac/--drop-in/--drop-seed/--dev-drop-*`
  (SH01 density), `--cand-topk/--cand-min-prob/--cand-ce-col/--cand-ce-min` (candidate cascade, recomputes population
  features on the pruned set, applied to TEST too for `--out`), `--prune-post` + `--comp-keep` (Shaunak's E023 regime,
  `density_mask` copy), `--extra-feats` (merges `<dir>/<split>_feats|_owner.parquet` by key), `--drop-cols`,
  `--test-frame`, `--fit-without-dropped`, `--lgb-params`. Tests in `tests/test_er_stage2.py`.
- `src/er_fullpass.py` — `--drop-s1-frac/--drop-in` in stage 2 + cache-key fix (from the SH01 period).
- `src/er_owner.py` — listwise owner model (prep/train/score; `--model`, `--epochs`, `--bf16`; slot-list leakage rule
  `exclude_fold0`). Tests `tests/test_er_owner.py`.
- `src/er_decide2.py` — exact expected-F (tested vs brute force, `tests/test_er_decide2.py`); no gain.
- `scripts/sm/*` (runner, bootstrap, push/pull code, ~45 job scripts), `scripts/gpu/*` (remote runner, links, put,
  post_file, bootstrap template, jobs gpu_main/ow03/ow04/status/to_tmux), `scripts/upload_to_shreyas.py`,
  `scripts/export_for_shreyas.py`, `scripts/githooks/pre-push`.
- Docs: `docs/SHREYAS_LOG.md` (full experiment log), `docs/SHREYAS_SAGEMAKER.md`, this handoff; CLAUDE.md top section.

## 9. Next steps (in order)

1. **Watch origin/shaunak for E034** (`git fetch origin; git log origin/shaunak -5`; his EXPERIMENTS row "E034").
   Chain prints "GATE PASS/FAIL". Report to the user immediately. (A background watcher in the old session cannot be
   relied on after a restart — poll manually.)
2. If **PASS**: final = `sub_E034_ow04_unseen` (same variant logic as E033); make sure his zip includes
   `src/er_owner.py`, the OW04 train/score commands (`scripts/gpu/jobs/ow04.sh`: e5-large, 2 epochs, batch 32, lr 2e-5,
   bf16, n 200000 → 137,362 groups) and the owner-model paragraph in the approach doc (prompt step 5).
   If **FAIL**: ship E033 unchanged.
3. Upload by ~20:00 IST (one final; keep a slot for fixes). Organiser validator must PASS on the uploaded pair.
4. After the deadline: **stop the SageMaker space** (cost); optionally clean S3 (`shreyas/`, `shreyas-gpu/`, `shared/`).
   Ask the user to revoke the GitHub PAT they pasted earlier (it was never used) and to change the lab password if reused.
5. If the user asks for more modelling: dev ceiling ~0.9911 (ambiguous empty-address contests); remaining levers were
   OW-style competitor-aware models; LLM 7B reranker deemed too slow on one GB10 for the time left (~7–8 prompts/s).

## 10. Private local files (contain presigned links; never commit or share publicly)

In `C:\Xhreyas\projects\amazon\`: `GPU_BOOTSTRAP_COMMAND.txt` (GPU box bootstrap, ~7 days), `SHAUNAK_UPLOAD_COMMAND.txt`
(POST uploader link, until ~3 Oct), `SHAUNAK_EXPORT_COMMAND.txt` (old, unused), `FEATURES_FOR_SHAUNAK.txt` (hand-off tars,
24 h from ~15:40 27 Sep), `FINAL_V1_LINKS.txt`, `FINAL_V2_LINKS.txt` (24 h), `LB_PROBE_LINKS.txt` (24 h),
`PROMPT_FOR_SHAUNAKS_CLAUDE.md` (no links). Regenerate links with boto3 `generate_presigned_url` (local `python` has boto3;
the team `.venv` does not).

## 11. Lessons / gotchas

- Bash tool mangles `\\n`/backslash-newline inside heredoc-embedded Python → use the Edit tool for such edits.
- Files written by Python on Windows get CRLF → `sed -i 's/\r$//'` before uploading shell scripts to S3.
- `add_extra` once globbed `train_*.parquet` and picked `train_groups.parquet` → load explicit file names.
- Restarting the GPU runner orphans (and on logout, kills) its jobs; the lab box kills SSH-session processes → tmux.
- Leakage rule for owner-model training must be based on the slot list the model sees (not full top-15 retrieval) —
  the stricter rule left only 48k easy groups.
- Dev regimes differ: normal dev vs our SH01 test-like dev (19% S1 drop) vs Shaunak's comp-keep 0.78 dev — compare only
  paired runs in the same regime.
- e5-large scoring on GB10 was ~370 pairs/s for CE pairs (much slower than estimated); owner scoring ~430→250 groups/s.
- The r7i heartbeat clock is UTC; S3 listings from the laptop show IST.
