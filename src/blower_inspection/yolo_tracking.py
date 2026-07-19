"""YOLO + ByteTrack part localisation and rotating-part inspection sessions."""

from __future__ import annotations

import importlib
import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackedPart:
    track_id: int
    bounds: tuple[int, int, int, int]
    confidence: float


class YoloByteTrackDetector:
    """Load a user supplied Ultralytics YOLO checkpoint and retain ByteTrack IDs."""

    def __init__(self, model_path: str | Path, confidence: float = 0.40) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"YOLO part detector checkpoint not found: {self.model_path}")
            if importlib.util.find_spec("ultralytics") is None:
                raise RuntimeError("Missing dependency 'ultralytics'. Install it with `pip install -e .[industrial]`.")
            self._model = importlib.import_module("ultralytics").YOLO(str(self.model_path))
        return self._model

    def track(self, frame: np.ndarray) -> list[TrackedPart]:
        """Detect parts and associate them with persistent ByteTrack IDs."""
        results = self._load().track(frame, persist=True, tracker="bytetrack.yaml", conf=self.confidence, verbose=False)
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        ids = boxes.id.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        height, width = frame.shape[:2]
        parts: list[TrackedPart] = []
        for box, track_id, confidence in zip(xyxy, ids, confidences):
            x0, y0, x1, y1 = box.astype(int)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width, x1), min(height, y1)
            if x1 > x0 and y1 > y0:
                parts.append(TrackedPart(int(track_id), (x0, y0, x1 - x0, y1 - y0), float(confidence)))
        return parts


@dataclass
class RotatingPartSession:
    track_id: int
    first_seen: float
    last_seen: float
    frames_inspected: int = 0
    has_failure: bool = False
    worst_score: float = 0.0


@dataclass(frozen=True)
class CompletedPart:
    track_id: int
    status: str
    frames_inspected: int
    worst_score: float


class RotatingPartInspector:
    """Aggregate all views belonging to a ByteTrack ID into one final verdict."""

    def __init__(self, lost_timeout_s: float = 1.0) -> None:
        self.lost_timeout_s = lost_timeout_s
        self.sessions: dict[int, RotatingPartSession] = {}

    def observe_tracks(self, tracks: list[TrackedPart], now: float | None = None) -> list[CompletedPart]:
        now = time.monotonic() if now is None else now
        for track in tracks:
            session = self.sessions.get(track.track_id)
            if session is None:
                self.sessions[track.track_id] = RotatingPartSession(track.track_id, now, now)
            else:
                session.last_seen = now
        return self._complete_lost(now, {track.track_id for track in tracks})

    def record_inspection(self, track_id: int, *, is_pass: bool, anomaly_score: float) -> None:
        session = self.sessions.get(track_id)
        if session is None:
            return
        session.frames_inspected += 1
        session.has_failure |= not is_pass
        session.worst_score = max(session.worst_score, anomaly_score)

    def needs_initial_inspection(self, track_id: int) -> bool:
        """Prefer a first view for every simultaneously tracked part."""
        session = self.sessions.get(track_id)
        return session is not None and session.frames_inspected == 0

    def flush(self) -> list[CompletedPart]:
        completed = [self._complete(session) for session in self.sessions.values() if session.frames_inspected]
        self.sessions.clear()
        return completed

    def _complete_lost(self, now: float, visible_ids: set[int]) -> list[CompletedPart]:
        expired = [track_id for track_id, session in self.sessions.items() if track_id not in visible_ids and now - session.last_seen >= self.lost_timeout_s]
        completed = [self._complete(self.sessions.pop(track_id)) for track_id in expired if self.sessions[track_id].frames_inspected]
        for track_id in expired:
            self.sessions.pop(track_id, None)
        return completed

    @staticmethod
    def _complete(session: RotatingPartSession) -> CompletedPart:
        return CompletedPart(session.track_id, "FAIL" if session.has_failure else "PASS", session.frames_inspected, session.worst_score)
