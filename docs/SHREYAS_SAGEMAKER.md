# Shreyas branch: SageMaker CPU box + S3 job queue

**Why this shape:** the IAM user `shreyas` can use S3 only (no SageMaker/EC2/IAM API), and the available SageMaker
space is `ml.r7i.8xlarge` (32 vCPU, 256 GB RAM, **no GPU**). So the GPU pipeline (embeddings, bi-encoder,
cross-encoders) stays on the team RTX 5080, and this box does RAM/CPU work on its stage-1/stage-2 outputs:
France adaptation, decision-layer experiments, extra LightGBM stages.

Bucket: `s3://amazon-sagemaker-548171706001-ap-southeast-2-bjmz86wfr8l6oy/shreyas/` (region ap-southeast-2).

## One-time setup (user, in the space's terminal)
Space settings: instance `ml.r7i.8xlarge`, storage ≥ 100 GB, idle shutdown off (or long).
```bash
aws s3 cp s3://amazon-sagemaker-548171706001-ap-southeast-2-bjmz86wfr8l6oy/shreyas/bootstrap.sh - | bash
```
`scripts/sm/bootstrap.sh`: clones the repo (`shreyas`), makes `.venv` from `requirements-core.txt`, unpacks the
organiser data to `data/dataset/`, starts `scripts/sm/runner.py` in the background.

## Queue protocol (`scripts/sm/runner.py`)
| S3 key under `shreyas/` | meaning |
|---|---|
| `queue/<job>.sh` | bash script to run in the repo root (picked up within 20 s, several run in parallel) |
| `logs/<job>.log` | stdout+stderr, synced every 60 s |
| `done/<job>.json` | exit code + runtime |
| `heartbeat.json` | every 60 s: RAM, load, disk, running jobs |
| `control/stop` | runner exits |
| `artifacts/...` | job outputs pushed back with `aws s3 cp` |

Jobs start with `git pull` when they need new code. The box never pushes to git.

## Artefacts from the GPU box
`scripts/export_for_shreyas.py` (run by the GPU-box owner) packs dev rows of the stage-1 train chunks, all test
chunks, CE scores and stage-2 outputs, then uploads tar parts to presigned S3 URLs under `shreyas/import/`.
No embeddings or model weights are moved.
