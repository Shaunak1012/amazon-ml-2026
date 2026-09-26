"""Submit a SageMaker training job (managed Spot, ml.g5.2xlarge by default) from the Studio space.

Runs on the r7i space via the S3 job queue, using the space's execution role. Spot jobs wait up to --max-wait for
capacity instead of failing, and checkpoint to S3 so an interruption resumes. Blocks until the job ends and prints
its status and log tail.

    python scripts/sm/gpu/launch.py --entry probe.py --name probe --max-run 900
"""
import argparse
import time

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

ap = argparse.ArgumentParser()
ap.add_argument("--entry", required=True, help="script inside scripts/sm/gpu/")
ap.add_argument("--name", required=True)
ap.add_argument("--instance", default="ml.g5.2xlarge")
ap.add_argument("--max-run", type=int, default=6 * 3600)
ap.add_argument("--max-wait", type=int, default=10 * 3600, help="spot: run time + time allowed to wait for capacity")
ap.add_argument("--on-demand", action="store_true")
ap.add_argument("--bucket", required=True)
ap.add_argument("--hp", nargs="*", default=[], help="key=value hyperparameters passed to the entry script")
a = ap.parse_args()

sess = sagemaker.Session(default_bucket=a.bucket)
role = sagemaker.get_execution_role(sess)
job = f"shreyas-{a.name}-{time.strftime('%m%d-%H%M%S')}"
est = PyTorch(
    entry_point=a.entry, source_dir="scripts/sm/gpu", role=role, instance_type=a.instance, instance_count=1,
    framework_version="2.3", py_version="py311", sagemaker_session=sess, max_run=a.max_run,
    use_spot_instances=not a.on_demand, max_wait=None if a.on_demand else max(a.max_wait, a.max_run),
    checkpoint_s3_uri=f"s3://{a.bucket}/shreyas/gpu/ckpt/{a.name}",
    output_path=f"s3://{a.bucket}/shreyas/gpu/output", code_location=f"s3://{a.bucket}/shreyas/gpu/code",
    hyperparameters=dict(kv.split("=", 1) for kv in a.hp), keep_alive_period_in_seconds=0,
    environment={"S3_BUCKET": a.bucket},
)
print(f"role {role}\nsubmitting {job} on {a.instance} ({'on-demand' if a.on_demand else 'spot'})", flush=True)
est.fit(job_name=job, wait=False)
sm = boto3.client("sagemaker")
last = ""
while True:
    d = sm.describe_training_job(TrainingJobName=job)
    s = f"{d['TrainingJobStatus']} / {d.get('SecondaryStatus')}"
    if s != last:
        print(time.strftime("%H:%M:%S"), s, d.get("FailureReason", ""), flush=True)
        last = s
    if d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
        break
    time.sleep(30)
logs = boto3.client("logs")
try:
    streams = logs.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=job)["logStreams"]
    for st in streams:
        ev = logs.get_log_events(logGroupName="/aws/sagemaker/TrainingJobs", logStreamName=st["logStreamName"],
                                 startFromHead=False, limit=60)["events"]
        print("\n".join(e["message"] for e in ev))
except Exception as e:  # noqa: BLE001  (logs are a convenience; the status above is what matters)
    print("log fetch failed:", e)
print("BILLABLE_S", d.get("BillableTimeInSeconds"), "STATUS", d["TrainingJobStatus"])
