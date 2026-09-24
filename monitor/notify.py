"""Notifications: Discord webhook (primary) + ntfy (optional fallback).

Secrets come from .env (DISCORD_WEBHOOK_URL, NTFY_TOPIC_URL). The URL is never printed
or logged; errors report only the HTTP status.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

RED, YELLOW, GREEN, BLUE = 0xE74C3C, 0xF1C40F, 0x2ECC71, 0x3498DB
LEVEL_COLOR = {"critical": RED, "warning": YELLOW, "success": GREEN, "info": BLUE}
_UA = "DiscordBot (https://github.com, 1.0) amazon-ml-monitor"


def load_dotenv_once() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def _redact(text: str) -> str:
    return re.sub(r"https?://\S+", "<url>", text)


@dataclass
class Notifier:
    """send(level, title, fields) -> fan out to configured channels.

    dry_run=True records messages in .sent instead of sending (tests/demo).
    """

    discord_url: str | None = None
    ntfy_url: str | None = None
    dry_run: bool = False
    tag: str = ""
    sent: list[dict] = field(default_factory=list)

    @classmethod
    def from_env(cls, dry_run: bool = False) -> Notifier:
        load_dotenv_once()
        return cls(
            discord_url=os.environ.get("DISCORD_WEBHOOK_URL") or None,
            ntfy_url=os.environ.get("NTFY_TOPIC_URL") or None,
            dry_run=dry_run,
            tag=os.environ.get("OWNER", ""),
        )

    @property
    def channels(self) -> list[str]:
        return [n for n, u in (("discord", self.discord_url), ("ntfy", self.ntfy_url)) if u]

    def send(self, level: str, title: str, description: str = "", fields: dict | None = None,
             mention: bool | None = None) -> dict[str, str]:
        """Returns {channel: "ok" | "error: ..." | "dry-run"}."""
        mention = (level == "critical") if mention is None else mention
        msg = {"level": level, "title": title, "description": description, "fields": fields or {}, "mention": mention}
        self.sent.append(msg)
        if self.dry_run:
            return {"dry-run": "ok"}
        results: dict[str, str] = {}
        if self.discord_url:
            results["discord"] = self._discord(msg)
        # ntfy: fallback if Discord failed/missing, or always for critical alerts
        if self.ntfy_url and (results.get("discord") != "ok" or level == "critical"):
            results["ntfy"] = self._ntfy(msg)
        if not results:
            results["none"] = "no channels configured (set DISCORD_WEBHOOK_URL / NTFY_TOPIC_URL in .env)"
        return results

    # -------------------------------------------------------------- channels
    def _discord(self, m: dict) -> str:
        embed = {
            "title": m["title"][:256],
            "description": m["description"][:4000],
            "color": LEVEL_COLOR.get(m["level"], BLUE),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fields": [{"name": str(k)[:256], "value": str(v)[:1024] or "-", "inline": len(str(v)) < 40}
                       for k, v in list(m["fields"].items())[:25]],
        }
        if self.tag:
            embed["footer"] = {"text": self.tag}
        payload = {
            "content": "@here" if m["mention"] else "",
            "embeds": [embed],
            "allowed_mentions": {"parse": ["everyone"] if m["mention"] else []},
        }
        req = urllib.request.Request(
            self.discord_url, data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", "User-Agent": _UA},
        )
        return self._do(req)

    def _ntfy(self, m: dict) -> str:
        body = m["description"] + "\n" + "\n".join(f"{k}: {v}" for k, v in m["fields"].items())
        prio = {"critical": "urgent", "warning": "high"}.get(m["level"], "default")
        title = m["title"].encode("ascii", "ignore").decode() or "training monitor"
        req = urllib.request.Request(
            self.ntfy_url, data=body.encode("utf-8")[:4000], method="POST",
            headers={"Title": title[:200], "Priority": prio, "Tags": m["level"], "User-Agent": _UA},
        )
        return self._do(req)

    @staticmethod
    def _do(req: urllib.request.Request) -> str:
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return "ok" if 200 <= r.status < 300 else f"error: HTTP {r.status}"
        except urllib.error.HTTPError as e:
            return f"error: HTTP {e.code}"
        except Exception as e:  # never leak the URL via exception text
            return f"error: {type(e).__name__}: {_redact(str(e))}"
