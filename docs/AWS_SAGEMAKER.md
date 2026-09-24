# AWS SageMaker AI — practical guide

> Credit amounts, instance quotas, and prices change — **verify in AWS console** (Billing → Credits;
> Service Quotas → SageMaker). Nothing here assumes a specific credit balance.

## What it is
Amazon SageMaker AI is AWS's managed ML platform. The pieces that matter for us:
- **Studio / JupyterLab spaces** (or classic notebook instances): a hosted notebook on a GPU instance you pick.
- **Training jobs**: run a script on N instances, write artifacts to S3, instance shuts down automatically when done.
- **Endpoints**: hosted inference servers — we don't need these (batch-predict locally or in a training job).

## When to use it vs the local RTX 5080
| Use local 5080 (default) | Use SageMaker |
|---|---|
| Anything fitting 16 GB VRAM | Model needs > 16 GB (e.g. 7B+ bf16 LoRA, big VLM) → `ml.g6e` (L40S 48 GB) / `ml.p4d` (A100) |
| Iterative debugging | Parallel sweeps/seeds while the local GPU runs the main job |
| Data already local | A teammate without a GPU needs to run their own experiments |
Rule of thumb: only burst to AWS when the local queue is full for > 6 h or VRAM is the blocker. Upload/download
time and setup (~30–60 min first time) are real costs in a 72 h event.

## Quick path A — notebook (simplest)
1. Console → SageMaker AI → Studio → create/open domain (default settings) → JupyterLab space.
2. Pick instance: `ml.g5.xlarge` (A10G 24 GB) or `ml.g6.xlarge` (L4 24 GB) for most things; verify quota ≥ 1.
3. Clone the repo in the terminal, `pip install -r requirements-core.txt` + torch for the image's CUDA.
4. Copy data from S3: `aws s3 sync s3://<bucket>/data ./data`.
5. **Stop the space when idle** (auto-shutdown/idle timeout if available — verify in AWS console).

## Quick path B — training job (fire-and-forget, auto-stops)
```python
# pip install sagemaker ; run from a machine with AWS credentials (aws configure / SSO)
from sagemaker.pytorch import PyTorch
est = PyTorch(
    entry_point="src/train_template.py", source_dir=".",          # repo root; requirements.txt picked up
    role="<SageMakerExecutionRoleArn>",                             # verify in AWS console (IAM)
    instance_type="ml.g5.2xlarge", instance_count=1,
    framework_version="2.5", py_version="py311",                    # verify supported versions in console/docs
    hyperparameters={"cfg": "configs/E010-x.yaml"},
    max_run=6 * 3600,                                               # hard cap = cost cap
    use_spot_instances=True, max_wait=8 * 3600,                    # ~60-70% cheaper; needs checkpointing
    checkpoint_s3_uri="s3://<bucket>/ckpt/E010",                    # our scripts are resumable → spot-safe
)
est.fit({"train": "s3://<bucket>/data/"})
```
Inside the job, data is at `/opt/ml/input/data/train`, outputs go to `/opt/ml/model` → set `DATA_DIR`/`RUNS_DIR` accordingly.

## Cost hygiene (do all of these)
- **Budgets:** Billing → Budgets → create a monthly cost budget with email alerts at 50/80/100%.
- **Stop what you're not using:** Studio spaces/notebook instances bill while running, even idle.
- **Delete endpoints** you created (they bill 24/7): SageMaker → Inference → Endpoints.
- `max_run` on every training job; prefer spot + checkpoints.
- Delete large S3 artifacts after the event; check EBS volumes of stopped notebook instances.
- Region: pick one (e.g. `ap-south-1` Mumbai) and keep data + compute in it to avoid transfer fees.
- Before the Finale, check Billing → Bills for any still-running resources.

## Rules check
Before using AWS-hosted models (Bedrock/JumpStart), confirm COMPETITION.md allows the model and API usage.
