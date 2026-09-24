# Amazon ML Challenge 2026: team repository

Our team's entry for the Amazon ML Challenge 2026 (India, hosted on Unstop): a 72-hour machine learning
hackathon that opens on **25 Sep 2026 at 00:00 IST**. The problem statement and dataset have not been released
yet. Right now this repo is a **problem-agnostic scaffold**. It contains:
- an exact-metric library with tests,
- a shared cross-validation splitter,
- out-of-fold prediction storage for ensembling,
- a submission validator,
- a resumable GPU training template,
- a training heartbeat monitor that sends Discord alerts.

We'll replace this paragraph with a description of the approach once the task is known.

## Current best
| | Local CV | Public LB | Experiment | Tag |
|---|---|---|---|---|
| Best | — | — | — | — |

## Quick start
```bash
# 1. Setup (Windows: .venv\Scripts\activate · WSL/Linux: source .venv/bin/activate)
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt        # GPU stack (CUDA 12.8 wheels). Cloud/CPU-only: requirements-core.txt
copy .env.example .env                 # set DATA_DIR, DISCORD_WEBHOOK_URL (never commit .env)
python scripts/check_env.py            # verifies GPU (sm_120), bf16, libraries
python -m pytest                       # all tests run on synthetic data

# 2. Train (terminal 1) and monitor (terminal 2)
python -m monitor.launch --run E001-baseline -- python -m src.train_template --cfg configs/example.yaml
python -m monitor.watch --run E001-baseline

# 3. Predict → validate → tag → submit
python -m src.submission validate submissions/E001.csv --sample data/sample_submission.csv
python scripts/tag_submission.py submissions/E001.csv --exp E001-baseline --cv <cv> --sample data/sample_submission.csv

# 4. Final code zip (from committed files only)
python scripts/make_submission_zip.py --predictions <final.csv> --sample <sample.csv> --doc <approach.pdf>
```

## Repository structure
```
configs/    YAML configs: base.yaml + one per experiment (E###-name.yaml)
src/        metrics, CV folds, OOF storage, submission writer/validator, config, seeding, training template
monitor/    Heartbeat (import in training), watchdog dashboard + Discord/ntfy alerts, launcher, demo
scripts/    environment check, submission tagging, final zip builder
tests/      pytest suite (no GPU or real data needed)
notebooks/  EDA
docs/       competition facts, playbook, logs (see below)
```
`data/`, `runs/`, `checkpoints/`, `oof/`, `submissions/`, `dist/`, and `.env` are gitignored.

## Documentation
| Doc | Purpose |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Working rules for the team and for Claude Code |
| [docs/COMPETITION.md](docs/COMPETITION.md) | Rules, metric, data schema, limits, deadlines |
| [docs/PLAYBOOK_72H.md](docs/PLAYBOOK_72H.md) | Hour-by-hour plan and strategies for each problem type |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Every experiment: config, CV, LB, runtime |
| [docs/SUBMISSIONS.md](docs/SUBMISSIONS.md) | Submission ledger with git tags |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Key decisions and why we made them |
| [docs/DAILY_LOG.md](docs/DAILY_LOG.md) | Progress summaries and the GPU queue |
| [docs/SUBMISSION_CHECKLIST.md](docs/SUBMISSION_CHECKLIST.md) | Pre-submit checks and the approach-doc template |
| [docs/TEAM.md](docs/TEAM.md) | Roles, branches, how we share OOF predictions |
| [docs/HARDWARE.md](docs/HARDWARE.md) | RTX 5080 setup, limits, WSL2 |
| [docs/AWS_SAGEMAKER.md](docs/AWS_SAGEMAKER.md) | When and how to use AWS, and how to control cost |
| [monitor/README.md](monitor/README.md) | Training monitor and alerts |

## Reproducibility
Each leaderboard submission has a git tag (`sub-NN`) recorded in `docs/SUBMISSIONS.md`. Each experiment has
a config, a seed, and saved out-of-fold predictions. To rebuild the final submission, check out tag `final`
and follow Quick start.
