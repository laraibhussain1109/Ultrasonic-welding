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

    def exact_crop(self, frame: np.ndarray) -> np.ndarray:
        """Return the highest-confidence YOLO part crop for training/inference parity."""
        results = self._load().predict(frame, conf=self.confidence, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            raise ValueError("YOLO did not detect a part in the training image")
        boxes = results[0].boxes
        best = int(boxes.conf.argmax().item())
        x0, y0, x1, y1 = boxes.xyxy[best].detach().cpu().numpy().astype(int)
        height, width = frame.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("YOLO returned an empty part crop for a training image")
        return frame[y0:y1, x0:x1].copy()


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
    valid_views: int = 0
    invalid_registration_views: int = 0
    geometry_strong_views: int = 0
    tao_only_candidate_views: int = 0
    persistent_candidate_count: int = 0
    worst_geometry_score: float = 0.0
    worst_tao_score: float = 0.0
    reason_codes: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class CompletedPart:
    track_id: int
    status: str
    frames_inspected: int
    worst_score: float
    valid_views: int = 0
    reason_codes: tuple[str, ...] = ()


class RotatingPartInspector:
    """Aggregate rotating views until the configured physical completion event.

    ByteTrack IDs are implementation details and can change when a hand briefly
    occludes a rotating cylinder. New IDs that overlap a recent active detection
    are therefore attached to the same physical session. Conveyor mode finalizes
    at line crossing; fixed-nest mode finalizes after the required rotation views.
    """

    def __init__(
        self,
        lost_timeout_s: float = 1.0,
        counting_line_ratio: float = 0.45,
        counting_direction: str = "left_to_right",
        minimum_rotation_views: int = 1,
        weak_candidate_required_views: int = 2,
        completion_mode: str = "counting_line",
    ) -> None:
        if not 0.0 < counting_line_ratio < 1.0:
            raise ValueError("counting_line_ratio must be between 0 and 1")
        if counting_direction not in {"left_to_right", "right_to_left"}:
            raise ValueError("counting_direction must be 'left_to_right' or 'right_to_left'")
        if completion_mode not in {"counting_line", "minimum_views"}:
            raise ValueError("completion_mode must be 'counting_line' or 'minimum_views'")
        self.lost_timeout_s = lost_timeout_s
        self.counting_line_ratio = counting_line_ratio
        self.counting_direction = counting_direction
        self.minimum_rotation_views = max(1, int(minimum_rotation_views))
        self.weak_candidate_required_views = max(1, int(weak_candidate_required_views))
        self.completion_mode = completion_mode
        self.sessions: dict[int, RotatingPartSession] = {}
        self.track_to_part: dict[int, int] = {}
        self.completed_tracker_ids: set[int] = set()

    def observe_tracks(
        self, tracks: list[TrackedPart], frame_width: int, now: float | None = None
    ) -> None:
        now = time.monotonic() if now is None else now
        active_ids = {track.track_id for track in tracks}
        self.completed_tracker_ids.intersection_update(active_ids)
        for track in tracks:
            if track.track_id in self.completed_tracker_ids:
                continue
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
        self, track_id: int, *, is_pass: bool, anomaly_score: float,
        view_valid: bool = True, immediate_failure: bool | None = None,
        provisional_candidate: bool = False, geometry_score: float = 0.0,
        tao_score: float | None = None, reason_codes: tuple[str, ...] | list[str] = (),
    ) -> CompletedPart | None:
        session = self._session_for_track(track_id)
        if session is None:
            return None
        session.frames_inspected += 1
        if not view_valid:
            session.invalid_registration_views += 1
        else:
            session.valid_views += 1
        if geometry_score >= 1.0:
            session.geometry_strong_views += 1
        if provisional_candidate:
            session.tao_only_candidate_views += 1
            session.persistent_candidate_count += 1
        else:
            session.persistent_candidate_count = 0
        # Old callers preserve fail-latching. Hybrid callers explicitly label
        # severe versus provisional evidence.
        severe = (not is_pass) if immediate_failure is None else immediate_failure
        session.has_failure |= severe or session.persistent_candidate_count >= self.weak_candidate_required_views
        session.worst_score = max(session.worst_score, anomaly_score)
        session.worst_geometry_score = max(session.worst_geometry_score, geometry_score)
        session.worst_tao_score = max(session.worst_tao_score, tao_score if tao_score is not None else anomaly_score)
        session.reason_codes.update(reason_codes)
        completion_triggered = (
            self.completion_mode == "minimum_views" or session.crossed_counting_line
        )
        enough_valid = session.valid_views >= self.minimum_rotation_views
        quality_exhausted = session.frames_inspected >= self.minimum_rotation_views * 2
        if completion_triggered and (enough_valid or quality_exhausted):
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
        return bool(
            session
            and (self.completion_mode == "minimum_views" or session.crossed_counting_line)
            and session.valid_views < self.minimum_rotation_views
            and session.frames_inspected < self.minimum_rotation_views * 2
        )

    def accepts_inspection(self, track_id: int) -> bool:
        """Return false after a physical part is finalized until its track leaves."""
        return track_id not in self.completed_tracker_ids and self._session_for_track(track_id) is not None

    def view_progress(self, track_id: int) -> tuple[int, int, int]:
        """Return inspected, valid and required view counts for operator status."""
        session = self._session_for_track(track_id)
        if session is None:
            return 0, 0, self.minimum_rotation_views
        return session.frames_inspected, session.valid_views, self.minimum_rotation_views

    def flush(self) -> list[CompletedPart]:
        """Clear unfinished sessions without counting parts that never crossed."""
        completed = [
            self._complete(session)
            for session in self.sessions.values()
            if (self.completion_mode == "minimum_views" or session.crossed_counting_line)
            and (session.valid_views >= self.minimum_rotation_views
                 or session.frames_inspected >= self.minimum_rotation_views * 2)
        ]
        self.sessions.clear()
        self.track_to_part.clear()
        self.completed_tracker_ids.clear()
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
        if self.completion_mode == "counting_line" and not self._entry_side(center_ratio):
            return None
        session = RotatingPartSession(track.track_id, now, now, track.bounds, {track.track_id})
        self.sessions[session.part_id] = session
        return session.part_id

    def _finish(self, session: RotatingPartSession) -> CompletedPart:
        completed = self._complete(session)
        self.sessions.pop(session.part_id, None)
        for tracker_id in session.tracker_ids:
            self.track_to_part.pop(tracker_id, None)
            self.completed_tracker_ids.add(tracker_id)
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

    def _complete(self, session: RotatingPartSession) -> CompletedPart:
        # Crossing with too few registered views is a fail-closed quality fault.
        insufficient = session.valid_views < self.minimum_rotation_views
        reasons = set(session.reason_codes)
        if insufficient:
            reasons.add("INSUFFICIENT_VIEW_QUALITY")
        return CompletedPart(session.part_id, "FAIL" if session.has_failure or insufficient else "PASS",
                             session.frames_inspected, session.worst_score, session.valid_views, tuple(sorted(reasons)))
