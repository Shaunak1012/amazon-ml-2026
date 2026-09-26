"""Credential-free job runner for a remote GPU box (e.g. a lab machine reached over VPN + SSH by the user).

The box holds NO AWS keys. It only has:
  - a presigned GET URL for one job-script object (s3://.../shreyas-gpu/queue/current.sh), polled every 20 s;
    each new content (by sha256) runs once, as its own process group;
  - a presigned POST policy (json: url + fields) that can only write keys under shreyas-gpu/out/.
Both expire after 7 days. Logs, heartbeat and done markers go to shreyas-gpu/out/{logs,done}/ and heartbeat.json.

    nohup setsid python scripts/gpu/remote_runner.py --queue-url "$Q" --post post.json > runner.out 2>&1 &
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]


def post(spec: dict, key: str, body: bytes) -> None:
    """Upload bytes to shreyas-gpu/out/<key> via the POST policy."""
    fields = {**spec["fields"], "key": f"shreyas-gpu/out/{key}"}
    r = requests.post(spec["url"], data=fields, files={"file": (Path(key).name, body)}, timeout=600)
    r.raise_for_status()


def gpu_line() -> str:
    """One-line nvidia-smi summary (empty if unavailable)."""
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-url", required=True)
    ap.add_argument("--post", required=True, help="json file: presigned POST {url, fields} for shreyas-gpu/out/")
    a = ap.parse_args()
    spec = json.loads(Path(a.post).read_text())
    jobs, logs = ROOT / "runs" / "gpu_jobs", ROOT / "runs" / "gpu_logs"
    jobs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    seen_f = jobs / "seen.txt"
    seen = set(seen_f.read_text().split()) if seen_f.exists() else set()
    running: dict[str, tuple[subprocess.Popen, float]] = {}
    last = 0.0
    while True:
        try:
            r = requests.get(a.queue_url, timeout=60)
            if r.status_code == 200 and r.content.strip():
                h = hashlib.sha256(r.content).hexdigest()[:12]
                if h not in seen:
                    first = r.content.decode().splitlines()[1:2]
                    name = f"{time.strftime('%m%d-%H%M%S')}-{h}"
                    (jobs / f"{name}.sh").write_bytes(r.content)
                    seen.add(h)
                    seen_f.write_text("\n".join(sorted(seen)))
                    log = open(logs / f"{name}.log", "ab")
                    p = subprocess.Popen(["bash", str(jobs / f"{name}.sh")], cwd=ROOT, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         env={**os.environ, "JOB_ID": name, "POST_JSON": str(Path(a.post).resolve())})
                    running[name] = (p, time.time())
                    post(spec, f"logs/{name}.log", f"[runner] started {name} {first}\n".encode())
            for name, (p, t0) in list(running.items()):
                if p.poll() is not None:
                    post(spec, f"logs/{name}.log", (logs / f"{name}.log").read_bytes())
                    post(spec, f"done/{name}.json", json.dumps({"job": name, "returncode": p.returncode,
                                                                "runtime_s": round(time.time() - t0)}).encode())
                    del running[name]
            if time.time() - last > 60:
                for name in running:
                    post(spec, f"logs/{name}.log", (logs / f"{name}.log").read_bytes())
                du = shutil.disk_usage(ROOT)
                post(spec, "heartbeat.json", json.dumps({
                    "t": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": gpu_line(), "load": os.getloadavg(),
                    "disk_free_gb": round(du.free / 1e9, 1), "running": {n: round(time.time() - t) for n, (_, t) in running.items()},
                }).encode())
                last = time.time()
        except Exception as e:  # noqa: BLE001  (network blips must never kill the runner)
            print(time.strftime("%H:%M:%S"), "runner error:", e, flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
