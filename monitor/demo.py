"""Prove the monitor works: simulate runs (clean, OOM crash, NaN, hang, hard kill) and assert alerts fire.

    python -m monitor.demo                   # dry run: alerts captured locally, no network
    python -m monitor.demo --notify          # also send to Discord/ntfy from .env
    python -m monitor.demo --scenarios crash # subset

Each scenario is a real subprocess launched through monitor.launch (so log capture,
exit codes, and PID checks are exercised exactly as in real training) and watched by
a Watchdog with shortened thresholds. Exit code 0 only if every scenario passes.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from monitor.heartbeat import Heartbeat, default_runs_dir
from monitor.notify import Notifier
from monitor.watch import Thresholds, Watchdog

# scenario -> predicate over sent messages
EXPECT = {
    "ok": lambda s: any(m["level"] == "success" for m in s) and any(m["level"] == "info" for m in s),
    "crash": lambda s: any(m["level"] == "critical" and "OOM" in m["title"] and m["mention"] for m in s),
    "nan": lambda s: any(m["level"] == "critical" and "NaN" in m["title"] for m in s),
    "hang": lambda s: any("stale" in m["title"] for m in s),
    "die": lambda s: any(m["level"] == "critical" and ("code 3" in m["title"] or "died" in m["title"]) for m in s),
}


def child(scenario: str, steps: int, step_time: float) -> None:
    """Fake training loop."""
    run_id = os.environ["RUN_ID"]
    with Heartbeat(run_id, every_steps=5, every_seconds=1, total_steps=steps, higher_is_better=False,
                   metric_name="smape") as hb:
        for s in range(1, steps + 1):
            time.sleep(step_time)
            loss = 2.0 * math.exp(-s / 30) + 0.1
            if scenario == "nan" and s >= steps // 2:
                loss = float("nan")
            hb.step(s, loss=loss, epoch=s / steps * 2, lr=1e-3, samples=64)
            if s % 20 == 0:
                print(f"step {s} loss {loss:.4f}", flush=True)
                hb.val(40 - s * 0.1)
            if scenario == "crash" and s == steps // 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB (simulated by monitor.demo)")
            if scenario == "hang" and s == steps // 3:
                print("simulating a hang (deadlocked dataloader)...", flush=True)
                time.sleep(3600)
            if scenario == "die" and s == steps // 2:
                print("simulating hard kill (segfault / OOM-killer)", flush=True)
                os._exit(3)


def run_scenario(name: str, runs_dir: Path, notifier: Notifier, timeout: float = 90) -> tuple[bool, list[dict], float]:
    run_id = f"demo-{name}-{time.strftime('%H%M%S')}"
    cmd = [sys.executable, "-m", "monitor.launch", "--run", run_id, "--runs-dir", str(runs_dir), "--quiet", "--",
           sys.executable, "-m", "monitor.demo", "--child", "--scenario", name, "--steps", "60", "--step-time", "0.1"]
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    th = Thresholds(stale_min=0.1, summary_min=0.05, gpu_idle_min=1e9, disk_min_gb=0, realert_min=60)
    wd = Watchdog(run_id, runs_dir, th, notifier)
    killed = False
    while time.time() - t0 < timeout:
        wd.check()
        if wd.done:
            break
        if name == "hang" and not killed and any("stale" in m["title"] for m in notifier.sent):
            _kill_tree(proc.pid)  # demo cleanup; watchdog should then report the exit
            killed = True
        time.sleep(0.5)
    if proc.poll() is None:
        _kill_tree(proc.pid)
    ok = EXPECT[name](notifier.sent)
    return ok, list(notifier.sent), time.time() - t0


def _kill_tree(pid: int) -> None:
    import psutil

    try:
        p = psutil.Process(pid)
        for c in p.children(recursive=True):
            c.kill()
        p.kill()
    except psutil.NoSuchProcess:
        pass


def _utf8_stdout() -> None:
    """Windows consoles/pipes default to cp1252; emoji in titles would crash print()."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m monitor.demo")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--scenario", default="ok", help=argparse.SUPPRESS)
    ap.add_argument("--steps", type=int, default=60, help=argparse.SUPPRESS)
    ap.add_argument("--step-time", type=float, default=0.1, help=argparse.SUPPRESS)
    ap.add_argument("--scenarios", default="ok,crash,nan,hang,die")
    ap.add_argument("--notify", action="store_true", help="send real notifications (Discord/ntfy from .env)")
    a = ap.parse_args()
    _utf8_stdout()
    if a.child:
        child(a.scenario, a.steps, a.step_time)
        return 0

    runs_dir = default_runs_dir() / "_demo"
    results = []
    for name in a.scenarios.split(","):
        base = Notifier.from_env(dry_run=not a.notify)
        print(f"--- scenario {name!r} ({'LIVE notify: ' + ','.join(base.channels) if a.notify else 'dry-run'})")
        ok, sent, dt = run_scenario(name, runs_dir, base)
        for m in sent:
            print(f"    [{m['level']:8}] {'@here ' if m['mention'] else ''}{m['title']}")
        results.append((name, ok, dt))
    print("\nscenario  result  seconds")
    for name, ok, dt in results:
        print(f"{name:9} {'PASS' if ok else 'FAIL':6}  {dt:5.1f}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
