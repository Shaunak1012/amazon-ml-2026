import json
import math
import os

import pytest

from monitor.heartbeat import Heartbeat
from monitor.notify import Notifier
from monitor.watch import Thresholds, Watchdog


def _wd(tmp_path, run="r1", **th):
    return Watchdog(run, tmp_path, Thresholds(**th), Notifier(dry_run=True))


def test_heartbeat_writes_and_tracks_best(tmp_path):
    with Heartbeat("r1", tmp_path, every_steps=1, total_steps=10, higher_is_better=False) as hb:
        for s in range(1, 6):
            hb.step(s, loss=1.0 / s, samples=32)
        assert hb.val(0.5) and not hb.val(0.7) and hb.val(0.4)
    d = json.loads((tmp_path / "r1" / "heartbeat.json").read_text())
    assert d["status"] == "completed" and d["best_metric"] == 0.4 and d["step"] == 5
    assert d["pid"] == os.getpid() and d["eta_seconds"] is not None


def test_heartbeat_nan_flag_and_raise(tmp_path):
    hb = Heartbeat("n", tmp_path, every_steps=1000)
    hb.step(3, loss=float("nan"))
    d = json.loads((tmp_path / "n" / "heartbeat.json").read_text())
    assert d["nan"] and d["nan_step"] == 3
    hb2 = Heartbeat("n2", tmp_path, nan_action="raise")
    with pytest.raises(FloatingPointError):
        hb2.step(1, loss=math.inf)


def test_heartbeat_marks_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with Heartbeat("f", tmp_path):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    d = json.loads((tmp_path / "f" / "heartbeat.json").read_text())
    assert d["status"] == "failed" and "out of memory" in d["error"]


def test_watchdog_completed_sends_final(tmp_path):
    with Heartbeat("r1", tmp_path) as hb:
        hb.step(10, loss=0.3)
        hb.val(0.9)
    wd = _wd(tmp_path)
    wd.check()
    assert wd.done and wd.final_status == "completed"
    assert wd.notifier.sent[-1]["level"] == "success" and "0.9" in wd.notifier.sent[-1]["description"]


def test_watchdog_failed_is_critical_with_mention(tmp_path):
    with pytest.raises(RuntimeError):
        with Heartbeat("r1", tmp_path):
            raise RuntimeError("CUDA out of memory")
    wd = _wd(tmp_path)
    wd.check()
    wd.check()  # dedup: still one alert
    crit = [m for m in wd.notifier.sent if m["level"] == "critical"]
    assert len(crit) == 1 and crit[0]["mention"] and "OOM" in crit[0]["title"]


def test_watchdog_nan_and_dead_pid(tmp_path):
    hb = Heartbeat("r1", tmp_path)
    hb.step(1, loss=float("nan"))
    wd = _wd(tmp_path, gpu_idle_min=1e9)
    wd.check()
    assert any("NaN" in m["title"] for m in wd.notifier.sent)
    # simulate a dead PID (e.g. hard kill): rewrite heartbeat with a non-existent pid
    p = tmp_path / "r1" / "heartbeat.json"
    d = json.loads(p.read_text())
    d["pid"] = 999_999_99
    p.write_text(json.dumps(d))
    wd.check()
    assert wd.done and wd.final_status == "died"


def test_watchdog_stale(tmp_path):
    hb = Heartbeat("r1", tmp_path)
    hb.step(1, loss=1.0)
    p = tmp_path / "r1" / "heartbeat.json"
    d = json.loads(p.read_text())
    d["unix_time"] -= 3600
    p.write_text(json.dumps(d))
    wd = _wd(tmp_path, stale_min=1, gpu_idle_min=1e9)
    wd.check()
    assert any("stale" in m["title"] for m in wd.notifier.sent)
    assert (tmp_path / "r1" / "watch.csv").exists()


def test_notifier_no_channels_reports_cleanly():
    res = Notifier().send("info", "t")
    assert "none" in res
