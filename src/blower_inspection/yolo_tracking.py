"""YOLO + ByteTrack localization and line-crossing inspection aggregation."""

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
    """Load a user supplied Ultralytics checkpoint and retain ByteTrack IDs."""

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
        """Run YOLO on every frame and associate detections with ByteTrack IDs."""
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
    part_id: int
    first_seen: float
    last_seen: float
    last_bounds: tuple[int, int, int, int]
    tracker_ids: set[int] = field(default_factory=set)
    frames_inspected: int = 0
    has_failure: bool = False
    worst_score: float = 0.0
    crossed_counting_line: bool = False


@dataclass(frozen=True)
class CompletedPart:
    track_id: int
    status: str
    frames_inspected: int
    worst_score: float


class RotatingPartInspector:
    """Latch every surface result until the physical part crosses a count line.

    ByteTrack IDs are implementation details and can change when a hand briefly
    occludes a rotating cylinder. New IDs that overlap a recent active detection
    are therefore attached to the same physical session. A session is finalized
    only after its center crosses the configured line in the configured direction.
    """

    def __init__(
        self,
        lost_timeout_s: float = 1.0,
        counting_line_ratio: float = 0.80,
        counting_direction: str = "left_to_right",
    ) -> None:
        if not 0.0 < counting_line_ratio < 1.0:
            raise ValueError("counting_line_ratio must be between 0 and 1")
        if counting_direction not in {"left_to_right", "right_to_left"}:
            raise ValueError("counting_direction must be 'left_to_right' or 'right_to_left'")
        self.lost_timeout_s = lost_timeout_s
        self.counting_line_ratio = counting_line_ratio
        self.counting_direction = counting_direction
        self.sessions: dict[int, RotatingPartSession] = {}
        self.track_to_part: dict[int, int] = {}

    def observe_tracks(
        self, tracks: list[TrackedPart], frame_width: int, now: float | None = None
    ) -> None:
        now = time.monotonic() if now is None else now
        for track in tracks:
            center_ratio = (track.bounds[0] + track.bounds[2] / 2.0) / max(frame_width, 1)
            part_id = self.track_to_part.get(track.track_id)
            if part_id is None:
                part_id = self._reattach_or_start(track, center_ratio, now)
                if part_id is None:
                    continue  # First observed beyond the line: it was not counted here.
            session = self.sessions[part_id]
            previous_center = (session.last_bounds[0] + session.last_bounds[2] / 2.0) / max(frame_width, 1)
            session.last_seen = now
            session.last_bounds = track.bounds
            session.tracker_ids.add(track.track_id)
            self.track_to_part[track.track_id] = part_id
            if self._crossed(previous_center, center_ratio):
                session.crossed_counting_line = True

    def record_inspection(
        self, track_id: int, *, is_pass: bool, anomaly_score: float
    ) -> CompletedPart | None:
        session = self._session_for_track(track_id)
        if session is None:
            return None
        session.frames_inspected += 1
        session.has_failure |= not is_pass
        session.worst_score = max(session.worst_score, anomaly_score)
        if session.crossed_counting_line:
            return self._finish(session)
        return None

    def latched_failure(self, track_id: int) -> bool:
        session = self._session_for_track(track_id)
        return bool(session and session.has_failure)

    def needs_initial_inspection(self, track_id: int) -> bool:
        session = self._session_for_track(track_id)
        return session is not None and session.frames_inspected == 0

    def needs_completion_inspection(self, track_id: int) -> bool:
        """Return whether this track crossed and needs its final cropped view."""
        session = self._session_for_track(track_id)
        return bool(session and session.crossed_counting_line)

    def flush(self) -> list[CompletedPart]:
        """Clear unfinished sessions without counting parts that never crossed."""
        completed = [self._complete(s) for s in self.sessions.values() if s.crossed_counting_line and s.frames_inspected]
        self.sessions.clear()
        self.track_to_part.clear()
        return completed

    def _reattach_or_start(self, track: TrackedPart, center_ratio: float, now: float) -> int | None:
        candidates = [
            session for session in self.sessions.values()
            if not session.crossed_counting_line
            and self._iou(session.last_bounds, track.bounds) >= 0.30
        ]
        if candidates:
            session = max(candidates, key=lambda item: self._iou(item.last_bounds, track.bounds))
            return session.part_id
        if not self._entry_side(center_ratio):
            return None
        session = RotatingPartSession(track.track_id, now, now, track.bounds, {track.track_id})
        self.sessions[session.part_id] = session
        return session.part_id

    def _finish(self, session: RotatingPartSession) -> CompletedPart:
        completed = self._complete(session)
        self.sessions.pop(session.part_id, None)
        for tracker_id in session.tracker_ids:
            self.track_to_part.pop(tracker_id, None)
        return completed

    def _session_for_track(self, track_id: int) -> RotatingPartSession | None:
        part_id = self.track_to_part.get(track_id)
        return self.sessions.get(part_id) if part_id is not None else None

    def _entry_side(self, center: float) -> bool:
        return center < self.counting_line_ratio if self.counting_direction == "left_to_right" else center > self.counting_line_ratio

    def _crossed(self, previous: float, current: float) -> bool:
        if self.counting_direction == "left_to_right":
            return previous < self.counting_line_ratio <= current
        return previous > self.counting_line_ratio >= current

    @staticmethod
    def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        x0, y0, x1, y1 = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, x1 - x0) * max(0, y1 - y0)
        return intersection / max(aw * ah + bw * bh - intersection, 1)

    @staticmethod
    def _complete(session: RotatingPartSession) -> CompletedPart:
        return CompletedPart(session.part_id, "FAIL" if session.has_failure else "PASS", session.frames_inspected, session.worst_score)
