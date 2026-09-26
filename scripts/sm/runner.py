"""S3-driven job runner for the SageMaker JupyterLab space (IAM user `shreyas` has S3 only, no SageMaker API).

Local side drops shell scripts into s3://<bucket>/shreyas/queue/<job_id>.sh; this loop (running on the space)
picks them up, runs each with bash in the repo root (several may run at once), and mirrors state back to S3:
    shreyas/heartbeat.json      every 60 s: time, load, RAM, disk, running jobs
    shreyas/logs/<job_id>.log   stdout+stderr, re-uploaded every 60 s while running
    shreyas/done/<job_id>.json  exit code + timings when the job ends
Put anything at shreyas/control/stop to make the runner exit (running jobs keep going).

    nohup python scripts/sm/runner.py --bucket <bucket> > ~/runner.out 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[2]


def mem_info() -> dict:
    """Total/available RAM in GB from /proc/meminfo (no psutil dependency)."""
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable"):
            out[k] = round(int(v.split()[0]) / 1e6, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="shreyas")
    ap.add_argument("--poll", type=int, default=20)
    a = ap.parse_args()
    s3 = boto3.client("s3")
    P = a.prefix.rstrip("/")
    jobs_dir, logs_dir = ROOT / "runs" / "sm_jobs", ROOT / "runs" / "sm_logs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple[subprocess.Popen, float]] = {}
    last_sync = 0.0

    def put(key: str, body: str | bytes) -> None:
        s3.put_object(Bucket=a.bucket, Key=f"{P}/{key}", Body=body)

    while True:
        # stop switch
        if s3.list_objects_v2(Bucket=a.bucket, Prefix=f"{P}/control/stop").get("KeyCount", 0):
            put("heartbeat.json", json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "state": "runner stopped"}))
            return
        # new jobs
        for obj in s3.list_objects_v2(Bucket=a.bucket, Prefix=f"{P}/queue/").get("Contents", []):
            key = obj["Key"]
            job = Path(key).stem
            if not key.endswith(".sh") or job in running:
                continue
            script = jobs_dir / f"{job}.sh"
            s3.download_file(a.bucket, key, str(script))
            s3.delete_object(Bucket=a.bucket, Key=key)
            log = open(logs_dir / f"{job}.log", "ab")
            proc = subprocess.Popen(["bash", str(script)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "JOB_ID": job, "S3_BUCKET": a.bucket, "S3_PREFIX": P},
                                    start_new_session=True)
            running[job] = (proc, time.time())
            put(f"logs/{job}.log", f"[runner] started {job} pid {proc.pid}\n")
        # finished jobs
        for job, (proc, t0) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                s3.upload_file(str(logs_dir / f"{job}.log"), a.bucket, f"{P}/logs/{job}.log")
                put(f"done/{job}.json", json.dumps({"job": job, "returncode": rc, "runtime_s": round(time.time() - t0),
                                                    "ended": time.strftime("%Y-%m-%d %H:%M:%S")}))
                del running[job]
        # periodic log sync + heartbeat
        if time.time() - last_sync > 60:
            for job in running:
                s3.upload_file(str(logs_dir / f"{job}.log"), a.bucket, f"{P}/logs/{job}.log")
            du = shutil.disk_usage(ROOT)
            put("heartbeat.json", json.dumps({
                "t": time.strftime("%Y-%m-%d %H:%M:%S"), "load": os.getloadavg(), **mem_info(),
                "disk_free_gb": round(du.free / 1e9, 1), "cpus": os.cpu_count(),
                "running": {j: round(time.time() - t0) for j, (_, t0) in running.items()}}))
            last_sync = time.time()
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
