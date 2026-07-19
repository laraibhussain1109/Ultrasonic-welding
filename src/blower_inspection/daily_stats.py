"""Persistent production counters using 07:00-to-07:00 operating days."""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from pathlib import Path


def operating_day(now: datetime | None = None) -> str:
    now = now or datetime.now()
    if now.time() < time(7, 0):
        now -= timedelta(days=1)
    return now.date().isoformat()


class DailyStatistics:
    def __init__(self, path: str | Path = "data/results/daily_statistics.json") -> None:
        self.path = Path(path)

    def counts(self, now: datetime | None = None) -> dict[str, int]:
        key = operating_day(now)
        data = self._load()
        return {name: int(data.get(key, {}).get(name, 0)) for name in ("inspected", "passed", "failed")}

    def record(self, status: str, now: datetime | None = None) -> dict[str, int]:
        if status not in {"PASS", "FAIL"}:
            raise ValueError(f"Cannot record non-final inspection status: {status}")
        key = operating_day(now)
        data = self._load()
        counts = data.setdefault(key, {"inspected": 0, "passed": 0, "failed": 0})
        counts["inspected"] += 1
        counts["passed" if status == "PASS" else "failed"] += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return {name: int(counts[name]) for name in ("inspected", "passed", "failed")}

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
