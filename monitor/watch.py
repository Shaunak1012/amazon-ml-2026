"""Watchdog: run in a separate terminal next to training.

    python -m monitor.watch --run E007-deberta-f0
    python -m monitor.watch --test-notify              # send a test Discord/ntfy message
    python -m monitor.watch --run E007 --once          # single check, print JSON, exit

Checks: heartbeat freshness, PID alive, exit code (monitor.launch), log tracebacks/OOM,
NaN/diverging loss, GPU idle / temperature, RAM, disk. Live `rich` dashboard +
runs/<run>/watch.csv. Critical alerts -> red Discord embed with @here; summaries every
--summary-min (default 20) as compact embeds; final message with best score on completion.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from monitor.heartbeat import default_runs_dir, read_json
from monitor.notify import Notifier

LOG_PATTERNS = [
    ("oom", re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|MemoryError", re.I)),
    ("traceback", re.compile(r"^Traceback \(most recent call last\)", re.M)),
    ("cuda_error", re.compile(r"CUDA error|device-side assert|illegal memory access|NCCL error", re.I)),
    ("killed", re.compile(r"^Killed$", re.M)),
]


# ------------------------------------------------------------------ system
class GPU:
    def __init__(self, index: int = 0) -> None:
        self.ok = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(self.h)
            self.name = name.decode() if isinstance(name, bytes) else name
            self.ok = True
        except Exception as e:  # no NVIDIA driver (cloud VM / CI)
            self.err = f"{type(e).__name__}: {e}"

    def sample(self) -> dict[str, Any]:
        if not self.ok:
            return {}
        nv, h = self.nv, self.h
        out: dict[str, Any] = {}
        for key, fn in (
            ("gpu_util", lambda: nv.nvmlDeviceGetUtilizationRates(h).gpu),
            ("vram_used_gb", lambda: round(nv.nvmlDeviceGetMemoryInfo(h).used / 2**30, 2)),
            ("vram_total_gb", lambda: round(nv.nvmlDeviceGetMemoryInfo(h).total / 2**30, 2)),
            ("gpu_temp_c", lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
            ("gpu_power_w", lambda: round(nv.nvmlDeviceGetPowerUsage(h) / 1000, 1)),
        ):
            try:
                out[key] = fn()
            except Exception:
                out[key] = None
        return out


def sample_system(path: Path) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "ram_pct": vm.percent,
        "ram_used_gb": round(vm.used / 2**30, 1),
        "disk_free_gb": round(shutil.disk_usage(path).free / 2**30, 1),
    }


def pid_alive(pid: int | None) -> bool | None:
    if not pid:
        return None
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return psutil.pid_exists(pid)


def fmt_secs(s: float | None) -> str:
    if s is None:
        return "-"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


# --------------------------------------------------------------- watchdog
@dataclass
class Thresholds:
    stale_min: float = 10.0
    gpu_idle_min: float = 10.0
    gpu_idle_util: float = 5.0
    temp_c: float = 85.0
    disk_min_gb: float = 20.0
    ram_max_pct: float = 95.0
    diverge_factor: float = 3.0
    diverge_min_step: int = 100
    summary_min: float = 20.0
    realert_min: float = 30.0


@dataclass
class Watchdog:
    run_id: str
    runs_dir: Path = field(default_factory=default_runs_dir)
    th: Thresholds = field(default_factory=Thresholds)
    notifier: Notifier | None = None
    log_path: Path | None = None
    gpu_index: int = 0

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir)
        self.dir = self.runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(self.log_path) if self.log_path else self.dir / "train.log"
        self.notifier = self.notifier or Notifier.from_env()
        self.gpu = GPU(self.gpu_index)
        self.t_start = time.time()
        self.active: dict[str, float] = {}   # alert key -> last sent time
        self.history: deque = deque(maxlen=12)  # (time, level, title) for dashboard
        self.gpu_idle_since: float | None = None
        self.last_summary = time.time()
        self.log_offset = 0
        self.log_hits: list[str] = []
        self.done = False
        self.final_status: str | None = None
        psutil.cpu_percent(interval=None)  # prime

    # -------------------------------------------------------- alert helpers
    def _alert(self, key: str, level: str, title: str, desc: str = "", fields: dict | None = None,
               once: bool = False) -> None:
        now = time.time()
        last = self.active.get(key)
        if last is not None and (once or now - last < self.th.realert_min * 60):
            return
        self.active[key] = now
        self.history.append((time.strftime("%H:%M:%S"), level, title))
        self.notifier.send(level, f"[{self.run_id}] {title}", desc, fields)

    def _clear(self, key: str) -> None:
        self.active.pop(key, None)

    def _log_tail(self, n_lines: int = 25) -> str:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 16384))
                lines = f.read().decode("utf-8", "replace").splitlines()
            return "\n".join(lines[-n_lines:])
        except FileNotFoundError:
            return "(no log file)"

    def _scan_log(self) -> list[str]:
        """Incrementally scan new log bytes for crash signatures."""
        hits = []
        try:
            with open(self.log_path, "rb") as f:
                f.seek(self.log_offset)
                chunk = f.read()
                self.log_offset = f.tell()
        except FileNotFoundError:
            return hits
        text = chunk.decode("utf-8", "replace")
        for name, pat in LOG_PATTERNS:
            if pat.search(text):
                hits.append(name)
        return hits

    @staticmethod
    def _code(text: str, limit: int = 1500) -> str:
        return "```\n" + text[-limit:].replace("```", "'''") + "\n```"

    # --------------------------------------------------------------- check
    def check(self) -> dict[str, Any]:
        now = time.time()
        hb = read_json(self.dir / "heartbeat.json") or {}
        launch = read_json(self.dir / "launch.json") or {}
        exitj = read_json(self.dir / "exit.json")
        pid = hb.get("pid") or launch.get("pid")
        alive = pid_alive(pid)
        hb_age = now - hb["unix_time"] if hb.get("unix_time") else None
        status = hb.get("status", "waiting")
        gpu = self.gpu.sample()
        sysm = sample_system(self.runs_dir)
        new_hits = self._scan_log()
        self.log_hits = sorted(set(self.log_hits) | set(new_hits))

        snap = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status, "pid": pid, "pid_alive": alive,
                "hb_age_s": round(hb_age, 1) if hb_age is not None else None,
                **{k: hb.get(k) for k in ("step", "total_steps", "epoch", "loss", "val_metric", "best_metric",
                                          "lr", "samples_per_sec", "eta_seconds", "metric_name")},
                **gpu, **sysm, "log_flags": ",".join(self.log_hits)}
        fields = self._fields(snap)

        # ---- terminal states
        if status == "completed" and not self.done:
            self.done, self.final_status = True, "completed"
            self.notifier.send("success", f"[{self.run_id}] ✅ run completed",
                               f"Best {hb.get('metric_name', 'val')}: **{hb.get('best_metric')}** "
                               f"(step {hb.get('best_step')})", fields, mention=False)
            self.history.append((time.strftime("%H:%M:%S"), "success", "completed"))
        elif status in ("failed", "interrupted") and not self.done:
            self.done, self.final_status = True, status
            err = hb.get("error") or self._log_tail()
            oom = "oom" in self.log_hits or "out of memory" in err.lower()
            title = "💥 CUDA OOM — run failed" if oom else f"💥 run {status}"
            self._alert("terminal", "critical", title, self._code(err), fields, once=True)
        elif exitj is not None and not self.done:
            rc = exitj.get("returncode")
            self.done, self.final_status = True, f"exited rc={rc}"
            if rc != 0:
                self._alert("terminal", "critical", f"💥 process exited with code {rc}",
                            self._code(self._log_tail()), fields, once=True)
            else:
                self._alert("terminal", "warning", "process exited 0 but heartbeat never marked completed",
                            self._code(self._log_tail(10)), fields, once=True)
        elif alive is False and status == "running" and not self.done:
            self.done, self.final_status = True, "died"
            self._alert("terminal", "critical", "💀 training process died (PID gone)",
                        self._code(self._log_tail()), fields, once=True)

        if not self.done:
            # ---- liveness
            if not hb and now - self.t_start > self.th.stale_min * 60:
                self._alert("no_hb", "critical", f"no heartbeat after {self.th.stale_min:g} min",
                            f"Expected {self.dir / 'heartbeat.json'}", fields)
            if hb_age is not None and hb_age > self.th.stale_min * 60:
                self._alert("stale", "critical", f"⏸ heartbeat stale for {fmt_secs(hb_age)} (hang?)",
                            f"PID alive: {alive}", fields)
            elif hb_age is not None:
                self._clear("stale")
            # ---- log signatures while still running (e.g. worker crash, OOM caught & retried)
            for h in new_hits:
                self._alert(f"log_{h}", "critical", f"log shows {h}", self._code(self._log_tail()), fields)
            # ---- loss health
            if hb.get("nan"):
                self._alert("nan", "critical", f"❌ NaN/inf loss at step {hb.get('nan_step')}", "", fields)
            loss, lmin = hb.get("loss"), hb.get("loss_min")
            if (isinstance(loss, (int, float)) and isinstance(lmin, (int, float)) and lmin > 0
                    and (hb.get("step") or 0) >= self.th.diverge_min_step and loss > self.th.diverge_factor * lmin):
                self._alert("diverge", "warning", f"📈 loss diverging ({loss:.4g} vs min {lmin:.4g})", "", fields)
            # ---- GPU idle
            util = gpu.get("gpu_util")
            if util is not None and status == "running" and util < self.th.gpu_idle_util:
                self.gpu_idle_since = self.gpu_idle_since or now
                if now - self.gpu_idle_since > self.th.gpu_idle_min * 60:
                    self._alert("gpu_idle", "warning",
                                f"GPU util <{self.th.gpu_idle_util:g}% for {fmt_secs(now - self.gpu_idle_since)}",
                                "Dataloader bottleneck, CPU-only step, or hang?", fields)
            else:
                self.gpu_idle_since = None
                self._clear("gpu_idle")
            # ---- periodic summary
            if now - self.last_summary >= self.th.summary_min * 60 and hb:
                self.last_summary = now
                self.notifier.send("info", f"[{self.run_id}] status", "", fields, mention=False)

        # ---- resources (always)
        t = gpu.get("gpu_temp_c")
        if t is not None and t >= self.th.temp_c:
            self._alert("temp", "warning", f"🌡 GPU temp {t}°C", "", fields)
        if sysm["disk_free_gb"] < self.th.disk_min_gb:
            self._alert("disk", "warning", f"💾 low disk: {sysm['disk_free_gb']} GB free", "", fields)
        if sysm["ram_pct"] >= self.th.ram_max_pct:
            self._alert("ram", "warning", f"RAM {sysm['ram_pct']}% used", "", fields)

        self._append_csv(snap)
        return snap

    def _fields(self, s: dict) -> dict:
        prog = f"{s.get('step')}/{s.get('total_steps') or '?'}"
        f = {"step": prog, "epoch": s.get("epoch"), "loss": _fmt(s.get("loss")),
             s.get("metric_name") or "val": _fmt(s.get("val_metric")), "best": _fmt(s.get("best_metric")),
             "ETA": fmt_secs(s.get("eta_seconds")), "samples/s": s.get("samples_per_sec")}
        if s.get("gpu_util") is not None:
            f["GPU"] = (f"{s['gpu_util']}% | {s.get('vram_used_gb')}/{s.get('vram_total_gb')} GB | "
                        f"{s.get('gpu_temp_c')}°C | {s.get('gpu_power_w')} W")
        f["host"] = f"CPU {s['cpu_pct']}% | RAM {s['ram_pct']}% | disk {s['disk_free_gb']} GB free"
        return {k: v for k, v in f.items() if v is not None}

    def _append_csv(self, snap: dict) -> None:
        path = self.dir / "watch.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(snap.keys()), extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(snap)

    # ------------------------------------------------------------- display
    def render(self, snap: dict):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        color = {"completed": "green", "failed": "red", "interrupted": "red", "running": "cyan"}.get(
            snap["status"], "yellow")
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold")
        t.add_column()
        rows = [
            ("status", f"[{color}]{snap['status']}[/]  pid={snap['pid']} alive={snap['pid_alive']}"
                       f"  hb_age={snap['hb_age_s']}s"),
            ("progress", f"step {snap.get('step')}/{snap.get('total_steps') or '?'}  epoch {snap.get('epoch')}"
                         f"  ETA {fmt_secs(snap.get('eta_seconds'))}  {snap.get('samples_per_sec')} samples/s"),
            ("loss / val", f"{_fmt(snap.get('loss'))}  /  {_fmt(snap.get('val_metric'))}  "
                           f"(best {_fmt(snap.get('best_metric'))})  lr={_fmt(snap.get('lr'))}"),
        ]
        if snap.get("gpu_util") is not None:
            rows.append(("GPU", f"{snap['gpu_util']}%  VRAM {snap.get('vram_used_gb')}/{snap.get('vram_total_gb')} GB"
                                f"  {snap.get('gpu_temp_c')}°C  {snap.get('gpu_power_w')} W"))
        rows.append(("host", f"CPU {snap['cpu_pct']}%  RAM {snap['ram_pct']}%  disk {snap['disk_free_gb']} GB free"))
        if snap.get("log_flags"):
            rows.append(("log", f"[red]{snap['log_flags']}[/]"))
        for r in rows:
            t.add_row(*r)
        alerts = Table(show_header=False, box=None)
        for ts, lvl, title in list(self.history)[-6:]:
            c = {"critical": "red", "warning": "yellow", "success": "green"}.get(lvl, "blue")
            alerts.add_row(ts, f"[{c}]{lvl}[/]", title)
        return Panel(Group(t, Panel(alerts, title="recent alerts")),
                     title=f"watch: {self.run_id}", subtitle=f"channels: {','.join(self.notifier.channels) or 'none'}")

    def run(self, interval: float = 15.0, dashboard: bool = True, exit_on_finish: bool = True) -> str | None:
        if dashboard:
            from rich.live import Live

            with Live(refresh_per_second=1, screen=False) as live:
                while True:
                    snap = self.check()
                    live.update(self.render(snap))
                    if self.done and exit_on_finish:
                        break
                    time.sleep(interval)
        else:
            while True:
                snap = self.check()
                print(json.dumps(snap, default=str))
                if self.done and exit_on_finish:
                    break
                time.sleep(interval)
        return self.final_status


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.5g}"
    return "-" if v is None else str(v)


def _utf8_stdout() -> None:
    """Windows consoles/pipes default to cp1252; emoji in titles would crash print()."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m monitor.watch")
    ap.add_argument("--run", help="run id (folder under runs/)")
    ap.add_argument("--runs-dir")
    ap.add_argument("--log", help="log file to scan (default runs/<run>/train.log)")
    ap.add_argument("--interval", type=float, default=15, help="seconds between checks")
    ap.add_argument("--stale-min", type=float, default=10)
    ap.add_argument("--gpu-idle-min", type=float, default=10)
    ap.add_argument("--gpu-idle-util", type=float, default=5)
    ap.add_argument("--temp-c", type=float, default=85)
    ap.add_argument("--disk-min-gb", type=float, default=20)
    ap.add_argument("--summary-min", type=float, default=20)
    ap.add_argument("--diverge-factor", type=float, default=3.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-dashboard", action="store_true", help="print JSON lines instead of rich UI")
    ap.add_argument("--keep", action="store_true", help="keep watching after the run ends")
    ap.add_argument("--once", action="store_true", help="one check, print snapshot JSON, exit")
    ap.add_argument("--dry-run", action="store_true", help="don't send notifications")
    ap.add_argument("--test-notify", action="store_true", help="send a test message to all channels and exit")
    a = ap.parse_args(argv)
    _utf8_stdout()

    notifier = Notifier.from_env(dry_run=a.dry_run)
    if a.test_notify:
        res = notifier.send("info", "🔔 monitor test", "Test notification from `monitor.watch --test-notify`.",
                            {"channels": ",".join(notifier.channels) or "none"}, mention=False)
        res2 = notifier.send("critical", "🚨 monitor test (critical)", "This is what a failure alert looks like.",
                             {"example": "CUDA out of memory"})
        print("info:", res, "\ncritical:", res2)
        return 0 if all(v == "ok" for v in {**res, **res2}.values()) else 1
    if not a.run:
        ap.error("--run is required (or use --test-notify)")

    th = Thresholds(stale_min=a.stale_min, gpu_idle_min=a.gpu_idle_min, gpu_idle_util=a.gpu_idle_util,
                    temp_c=a.temp_c, disk_min_gb=a.disk_min_gb, summary_min=a.summary_min,
                    diverge_factor=a.diverge_factor)
    wd = Watchdog(a.run, Path(a.runs_dir) if a.runs_dir else default_runs_dir(), th, notifier,
                  Path(a.log) if a.log else None, a.gpu)
    if a.once:
        print(json.dumps(wd.check(), indent=2, default=str))
        return 0
    try:
        status = wd.run(a.interval, dashboard=not a.no_dashboard, exit_on_finish=not a.keep)
    except KeyboardInterrupt:
        return 0
    return 0 if status in (None, "completed", "exited rc=0") else 2


if __name__ == "__main__":
    sys.exit(main())
