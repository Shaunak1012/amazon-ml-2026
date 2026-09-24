# 72-hour playbook

Clock starts at data release (H0 = 25 Sep 2026 00:00 IST). Adjust ±, but never skip the gates (✅).

## H0–3 · Read, EDA, metric, CV
- Read the full statement + rules twice; fill `docs/COMPETITION.md` (Opus+high). List open questions.
- Download data; row counts, dtypes, id format, sha256. Look at 20 random rows + 20 hardest-looking rows.
- EDA: target distribution (skew → transform?), missingness, duplicates (exact + near) within train and
  across train/test, train/test shift (adversarial validation: if AUC > 0.6, CV must mimic the shift).
- Implement the metric exactly + edge-case tests. ✅ `pytest tests/test_metrics.py` green.
- CV design → `docs/DECISIONS.md`. Group by anything that leaks (same product/brand/image/seller/near-dup text).
  Write one folds file; everyone uses it. ✅ Validate the official sample submission file.

## H3–12 · Baselines + first submission
- B0: constant/median prediction → CV sanity (should equal naive LB score).
- B1: fast strong baseline (TF-IDF/char n-grams + ridge/GBDT; or frozen encoder embeddings + ridge/GBDT).
- ✅ First submission by ~H3–4 using B1: verifies format, CV↔LB correlation, pipeline end-to-end.
- Start the first long GPU job (e.g. embedding extraction for all rows) and keep the GPU busy from here on.
- Record CV vs LB for every submission; if they disagree, stop and diagnose (Opus+high) before optimizing further.

## H12–48 · Strong models + features
- Pretrained encoders + cheap heads first (cached embeddings → MLP/GBDT), then fine-tune the best family.
- Feature work driven by error analysis on OOF (worst residuals by segment), not by guessing.
- Diversity for the ensemble: different backbones, input views (text/image/tabular), losses, target transforms, seeds.
- Every model: all folds, OOF + test preds saved, EXPERIMENTS.md row. Kill ideas that don't beat CV by > fold std.
- Sleep in shifts; queue long jobs before sleeping with the watchdog on (Discord alerts).
- Submit only when CV improves meaningfully; keep ≥ 2 submissions for Day 3.

## H48–64 · Ensembling + final training
- Blend on OOF: weighted average (optimize weights on OOF, constrained ≥ 0) → ridge/GBDT stacker if it beats blend by > std.
- Hill-climbing / greedy selection across all saved OOFs; check blend weights are stable across folds.
- Decide per model: fold-average test preds vs full-data refit (refit only if fold models are clearly undertrained).
- Seed-average the final components if GPU time allows.
- Freeze feature code by H60.

## H64–72 · Freeze, reproduce, document, submit early
- Code freeze. Reproduce the final submission from a clean clone + `make_submission_zip.py` (cloud or teammate).
- Final submission(s) by **H68** at the latest — never in the last hour (portal load, timezone, format surprises).
- Approach doc (1–2 pages; template in SUBMISSION_CHECKLIST.md) — Opus+high. README final scores. Tag `final`.

---
## Strategy by problem type

### Tabular
- GBDT trio (LightGBM/XGBoost/CatBoost) with native categoricals; target/count encodings **inside folds**.
- Feature ideas: group aggregates, ratios, frequency encodings, text length/stats, date parts.
- Tune with Optuna on 1–2 folds, confirm on all. Blend GBDT + NN (MLP on quantile-transformed features).

### Text (classification/regression)
- Baselines: TF-IDF (word 1–2 + char 3–5) + ridge/LR; sentence embeddings (e5/bge/gte) + ridge/GBDT.
- Fine-tune DeBERTa-v3 (base → large) with layer-wise LR decay, mean pooling, 2–4 epochs, bf16.
- Extract structured signals with regex (quantities, units, pack size "pack of 6", brand) → features for GBDT.
- LLM (≤ 8B QLoRA) only if rules allow and the task needs reasoning; check inference time on full test.

### Image
- Frozen backbone embeddings (CLIP/SigLIP/DINOv2/ConvNeXt) + head first; fine-tune the best one after.
- Download/resize once with a robust async downloader (retries, timeouts, placeholder for dead URLs, log failures).
- TTA (hflip) cheaply at the end; resolution matters more than epochs for fine detail.

### Multimodal text + image (2025-style)
- Concatenate cached text + image embeddings + handcrafted features → GBDT/MLP. Usually the best time/score ratio.
- Then: fine-tune a text model and an image model separately, stack their OOFs; CLIP-style joint fine-tune last.
- Missing images: explicit indicator feature; never let a failed download silently become zeros without a flag.
- SMAPE/price-like: train on log1p(target), MAE/Huber in log space; evaluate SMAPE on inverse; check bias by price band.

### Extraction / OCR / VQA (2024-style)
- Pipeline: OCR (PaddleOCR/docTR/Tesseract) → candidate spans → regex + unit normalization → rule/ML ranker.
- VLM (Qwen2.5-VL / Florence-2 / etc., licence permitting) with constrained output format; fine-tune with LoRA on train.
- Post-process to the exact allowed units/format; empty prediction when unsure if FP costs as much as FN.
- Metric is exact-match: normalize numbers (e.g. "10.0" vs "10"), units, whitespace exactly as the grader expects.

### Ranking / retrieval
- Two-stage: candidate generation (BM25 + dense embeddings) → re-ranker (GBDT with LambdaRank or cross-encoder).
- CV grouped by query; metric computed per query then averaged. Hard-negative mining for the cross-encoder.

## Anti-patterns (don't)
- Tuning on public LB. Changing folds midway. Unlogged "quick tests". Submitting without validator.
- Full fine-tuning a big model before a cached-embedding baseline exists.
- Leaving the GPU idle overnight. Starting a 10 h run without a 20-min smoke test on a subset.
