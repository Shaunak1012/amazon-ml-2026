# Hardware & environment

| Part | Spec | Practical meaning |
|---|---|---|
| GPU | NVIDIA RTX 5080 (Blackwell, **sm_120**), 16 GB GDDR7 | bf16 + TF32 fast; needs **PyTorch built with CUDA 12.8+** (cu128 wheels or newer) |
| CPU | Ryzen 9 9950X3D, 16C/32T | GBDTs on CPU are fast; 12–16 DataLoader workers; image decode/resize in parallel |
| RAM | 64 GB | 75k–500k-row tabular/text fits easily; cache image tensors/embeddings to disk (npy/parquet) |
| OS | Windows 11 Pro (native, **WSL not installed**, **Smart App Control ON**) — detected 2026-09-24 | see below |

## Smart App Control (SAC) — turned OFF by the owner on 2026-09-25 (was blocking fresh DLLs)
Update: with SAC off, a clean venv from `requirements-core.txt` passes the suite (66 passed, 2 GPU-only skipped).
Pinned versions are kept anyway (they're verified). Re-enable SAC after the event (reversible on this build).

### History
Detected 2026-09-24: `torch 2.11.0+cu128` failed with *"An Application Control policy has blocked this file … torch.dll"*;
`pandas 3.0.x` was blocked the same way. **Working pins (verified):** `torch 2.8.0+cu128`, `pandas 2.3.3`,
`bitsandbytes 0.50.2` (8-bit AdamW OK natively). SAC blocks unsigned native code without cloud reputation, so brand-new
wheels fail while older, widely-downloaded ones may pass (pandas 2.3.3 loads fine). This is a Windows **security
setting** — the owner decides; Claude will not change it. Options, best first:
1. **Use WSL2 (recommended anyway).** Linux binaries inside WSL aren't subject to SAC, and you get working
   bitsandbytes/flash-attn/`torch.compile`. Setup below.
2. Pin older, high-reputation wheels on native Windows (what `requirements.txt` does). May break again on any upgrade.
3. Turn SAC off temporarily (Windows Security → App & browser control → Smart App Control → Off), and turn it
   back on after the event from the same page. Since the March/April 2026 cumulative updates this is reversible
   without a reset; the GPU box is on 25H2 build 26200.9457 (Sep 2026 updates), so it qualifies. **Fastest fix
   if a package gets blocked mid-competition.** Owner's decision; Claude won't change security settings.

## Verify the GPU stack
```bash
python scripts/check_env.py
```
Checks: capability `(12, 0)`, `sm_120` in `torch.cuda.get_arch_list()`, CUDA build ≥ 12.8, bf16 support,
fp32/bf16/fp16 matmul TFLOPS + correctness vs fp64, bf16 autocast backward, GBDT GPU training, NVML, `.env` keys.
Measured 2026-09-24 (torch 2.8.0+cu128, 4096² matmul): fp32 ≈ 26 TFLOPS (TF32 off), bf16 ≈ 64, fp16 ≈ 88.
Enable TF32 for fp32 parts: `torch.backends.cuda.matmul.allow_tf32 = True`.
Old builds (cu118/cu121/cu124) raise *"no kernel image is available for execution on the device"* on sm_120.

## WSL2 setup (one-time, ~20 min; owner runs these — they change system features)
```powershell
wsl --install -d Ubuntu-24.04      # admin PowerShell, reboot when asked
```
Then in Ubuntu:
```bash
sudo apt update && sudo apt install -y python3.11-venv git build-essential   # or python3-venv for 3.12
git clone https://github.com/<owner>/amazon-ml-2026 ~/amazon-ml-2026 && cd ~/amazon-ml-2026
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python scripts/check_env.py
```
- Keep data **inside the Linux filesystem** (`~/data`), not `/mnt/c/...` — cross-FS I/O is 5–10× slower.
- The Windows NVIDIA driver provides CUDA inside WSL; do **not** install a Linux NVIDIA driver.
- Give WSL enough RAM: `%UserProfile%\.wslconfig` → `[wsl2]` `memory=56GB` `processors=32`.
- All repo scripts are OS-agnostic (pathlib, no shell-specific code), so native Windows and WSL both work.

## Practical training limits (16 GB VRAM) — starting points, measure before trusting
| Workload | Setting that fits | Notes |
|---|---|---|
| Default precision | **bf16 autocast** (no GradScaler needed) | `torch.backends.cuda.matmul.allow_tf32 = True` for fp32 parts |
| DeBERTa-v3-base / e5-base, seq 256 | batch 32 | seq 512 → batch 16 |
| DeBERTa-v3-large, seq 256 | batch 8 + grad accum 4, gradient checkpointing | ~3–4× slower than base |
| ViT-B/16 / ConvNeXt-T @224 fine-tune | batch 64–128 | channels_last helps convnets |
| CLIP/SigLIP ViT-L embedding extraction | batch 256, `torch.inference_mode()` + fp16/bf16 | cache embeddings once, train heads many times |
| LLM ≤ 3B params | LoRA in bf16, grad checkpointing, batch 4–8 @ seq 512 | full FT of 1B barely fits with 8-bit Adam |
| LLM 7–8B | **QLoRA 4-bit** (bitsandbytes; imports natively, QLoRA not yet benchmarked here), batch 1–4 + accum, seq ≤ 1024 | ~10–13 GB; inference with vLLM (WSL) |
| VLM 2–3B (Qwen2.5-VL-3B etc.) | LoRA bf16, small image res, batch 1–2 + accum | check model licence/size rules first |

- **Gradient checkpointing** (`model.gradient_checkpointing_enable()`): ~30–40% slower, big activation savings.
- **Gradient accumulation** to reach effective batch 32–64; scale LR with effective batch.
- **DataLoader**: `num_workers=12–16`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`.
  On native Windows workers use spawn (slow start, everything must be picklable); WSL/Linux forks.
- Pre-tokenize text once and cache; pre-download/resize images to ~256–384px JPEG/WebP on local NVMe.
- **GBDTs**: XGBoost `device="cuda"`, CatBoost `task_type="GPU"`; LightGBM pip wheel is CPU (32 threads is fast)
  — `check_env.py` reports which GPU backends work.
- OOM playbook: halve batch → grad checkpointing → shorter seq/smaller images → 8-bit optimizer → LoRA.
- Watch temps/power with `python -m monitor.watch`; sustained > 85 °C → check case airflow.
