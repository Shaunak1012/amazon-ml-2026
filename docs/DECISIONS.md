# Decision log

Newest first. Each entry: what we chose, what else we considered, why — in plain language.

## 2026-09-24 — Repo scaffold before data release
- **Chose:** a problem-agnostic scaffold (metrics registry, shared folds, OOF storage, submission validator,
  heartbeat monitor with Discord alerts) and written playbooks, instead of pre-building a solution.
- **Alternatives:** guessing the task (e.g. re-building the 2025 price model) — rejected: wasted effort if the task changes.
- **Why:** the first 3 hours decide how fast we iterate for the next 69; tooling that prevents format errors
  and silent crashes is useful for any task.

## 2026-09-24 — Environment on this Windows machine
- **Chose:** Python 3.11 venv; pinned pandas 2.3.3 and torch 2.8.0+cu128 — both load under Smart App Control
  (torch 2.11 and pandas 3.0 were blocked). Verified sm_120, bf16, matmul. WSL2 recommended but optional now.
- **Alternatives:** latest wheels (blocked by Smart App Control); turning SAC off. That is reversible on this build
  (post-April-2026 updates), so it's the fallback if a needed package is ever blocked.
- **Why:** get a working CUDA stack without changing security settings.
