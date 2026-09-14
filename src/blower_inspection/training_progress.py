"""Shared real-time progress reporting for model training and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Callable


@dataclass(frozen=True)
class TrainingProgress:
    stage: str
    completed: int
    total: int
    elapsed_seconds: float

    @property
    def eta_seconds(self) -> float | None:
        if self.completed <= 0 or self.total <= self.completed:
            return 0.0 if self.total <= self.completed else None
        return self.elapsed_seconds * (self.total - self.completed) / self.completed

    def format(self) -> str:
        percent = 100.0 * self.completed / max(self.total, 1)
        elapsed = str(timedelta(seconds=int(self.elapsed_seconds)))
        eta = "calculating" if self.eta_seconds is None else str(timedelta(seconds=int(self.eta_seconds)))
        return f"{self.stage}: {self.completed}/{self.total} ({percent:.1f}%) | elapsed {elapsed} | ETA {eta}"


ProgressCallback = Callable[[TrainingProgress], None]
