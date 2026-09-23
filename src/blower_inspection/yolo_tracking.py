"""YOLO + ByteTrack localization and line-crossing inspection aggregation."""

from __future__ import annotations

import importlib
import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_COMPLETION_MODES = frozenset({"counting_line", "minimum_views", "part_departure"})


@dataclass(frozen=True)
class TrackedPart:
    track_id: int
    bounds: tuple[int, int, int, int]
    confidence: float


class YoloByteTrackDetector:
    """Load a user supplied Ultralytics checkpoint and retain ByteTrack IDs."""

    def __init__(self, model_path: str | Path, confidence: float = 0.70) -> None:
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
            if float(confidence) <= self.confidence:
                continue
            x0, y0, x1, y1 = box.astype(int)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width, x1), min(height, y1)
            if x1 > x0 and y1 > y0:
                parts.append(TrackedPart(int(track_id), (x0, y0, x1 - x0, y1 - y0), float(confidence)))
        return parts

    def detect_best(self, frame: np.ndarray) -> TrackedPart:
        """Return one qualified detection for operator-approved ROI locking.

        Unlike :meth:`track`, this does not require ByteTrack to have assigned
        an ID on the first camera frame. It is only used while inspection is
        stopped; production frames then reuse the approved coordinates.
        """
        results = self._load().predict(frame, conf=self.confidence, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            raise ValueError("YOLO did not detect a blower for ROI confirmation")
        boxes = results[0].boxes
        confidences = boxes.conf.detach().cpu().numpy()
        eligible = np.flatnonzero(confidences > self.confidence)
        if eligible.size == 0:
            raise ValueError(f"YOLO did not detect a blower above {self.confidence:.0%} confidence")
        best = int(eligible[np.argmax(confidences[eligible])])
        x0, y0, x1, y1 = boxes.xyxy[best].detach().cpu().numpy().astype(int)
        height, width = frame.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("YOLO returned an empty ROI")
        return TrackedPart(1, (x0, y0, x1 - x0, y1 - y0), float(confidences[best]))

    def exact_crop(self, frame: np.ndarray) -> np.ndarray:
        """Return the highest-confidence YOLO part crop for training/inference parity."""
        results = self._load().predict(frame, conf=self.confidence, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            raise ValueError("YOLO did not detect a part in the training image")
        boxes = results[0].boxes
        confidences = boxes.conf.detach().cpu().numpy()
        eligible = np.flatnonzero(confidences > self.confidence)
        if eligible.size == 0:
            raise ValueError(f"YOLO did not detect a part above {self.confidence:.0%} confidence")
        best = int(eligible[np.argmax(confidences[eligible])])
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
    blurry_views: int = 0
    glare_rejected_views: int = 0
    patchcore_candidate_views: int = 0
    geometry_candidate_views: int = 0
    confirmed_defect_views: int = 0
    last_candidate_sections: set[int] = field(default_factory=set)
    confirmed_defect_sections: set[int] = field(default_factory=set)
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
        counting_axis: str = "x",
    ) -> None:
        if not 0.0 < counting_line_ratio < 1.0:
            raise ValueError("counting_line_ratio must be between 0 and 1")
        if counting_axis not in {"x", "y"}:
            raise ValueError("counting_axis must be 'x' or 'y'")
        valid_directions = ({"left_to_right", "right_to_left"} if counting_axis == "x"
                            else {"top_to_bottom", "bottom_to_top"})
        if counting_direction not in valid_directions:
            raise ValueError(f"counting_direction {counting_direction!r} is invalid for axis {counting_axis!r}")
        if completion_mode not in SUPPORTED_COMPLETION_MODES:
            raise ValueError("completion_mode must be counting_line, minimum_views or part_departure")
        self.lost_timeout_s = lost_timeout_s
        self.counting_line_ratio = counting_line_ratio
        self.counting_direction = counting_direction
        self.counting_axis = counting_axis
        self.minimum_rotation_views = max(1, int(minimum_rotation_views))
        self.weak_candidate_required_views = max(1, int(weak_candidate_required_views))
        self.completion_mode = completion_mode
        self.sessions: dict[int, RotatingPartSession] = {}
        self.track_to_part: dict[int, int] = {}
        self.completed_tracker_ids: set[int] = set()
        # Fixed-nest sessions can complete while the rejected part is still on
        # the table. Retain its longitudinal locations until YOLO confirms that
        # tracker has left so the operator can rotate back to the defect.
        self.completed_defect_sections: dict[int, frozenset[int]] = {}

    def observe_tracks(
        self, tracks: list[TrackedPart], frame_width: int, now: float | None = None,
        frame_height: int | None = None,
    ) -> list[CompletedPart]:
        now = time.monotonic() if now is None else now
        active_ids = {track.track_id for track in tracks}
        self.completed_tracker_ids.intersection_update(active_ids)
        self.completed_defect_sections = {
            track_id: sections for track_id, sections in self.completed_defect_sections.items()
            if track_id in active_ids
        }
        for track in tracks:
            if track.track_id in self.completed_tracker_ids:
                continue
            center_ratio = self._center_ratio(track.bounds, frame_width, frame_height)
            part_id = self.track_to_part.get(track.track_id)
            if part_id is None:
                part_id = self._reattach_or_start(track, center_ratio, now)
                if part_id is None:
                    continue  # First observed beyond the line: it was not counted here.
            session = self.sessions[part_id]
            previous_center = self._center_ratio(session.last_bounds, frame_width, frame_height)
            session.last_seen = now
            session.last_bounds = track.bounds
            session.tracker_ids.add(track.track_id)
            self.track_to_part[track.track_id] = part_id
            if self._crossed(previous_center, center_ratio):
                session.crossed_counting_line = True

        # A fixed-nest inspection has no counting-line event. If the operator
        # removes a part before enough distinct views were acquired, terminate
        # that session after the configured loss debounce instead of leaving a
        # stale session (and its last annotated image) on screen indefinitely.
        # This is deliberately fail-closed: an incompletely inspected part is
        # reported as failed, never silently discarded as a pass.
        completed: list[CompletedPart] = []
        if self.completion_mode in {"minimum_views", "part_departure"}:
            lost = [
                session for session in self.sessions.values()
                if not (session.tracker_ids & active_ids)
                and now - session.last_seen >= self.lost_timeout_s
            ]
            for session in lost:
                completed.append(self._finish(session))
        return completed

    def record_inspection(
        self, track_id: int, *, is_pass: bool, anomaly_score: float,
        view_valid: bool = True, immediate_failure: bool | None = None,
        provisional_candidate: bool = False, geometry_score: float = 0.0,
        tao_score: float | None = None, reason_codes: tuple[str, ...] | list[str] = (),
        candidate_sections: tuple[int, ...] | list[int] = (),
    ) -> CompletedPart | None:
        session = self._session_for_track(track_id)
        if session is None:
            return None
        session.frames_inspected += 1
        if not view_valid:
            session.invalid_registration_views += 1
            if "MOTION_BLUR" in reason_codes or "LOW_SHARPNESS" in reason_codes:
                session.blurry_views += 1
        else:
            session.valid_views += 1
        if geometry_score >= 1.0:
            session.geometry_strong_views += 1
        if provisional_candidate:
            session.patchcore_candidate_views += 1
            sections = set(candidate_sections)
            # A repeated scalar spike is not persistence. Require the anomaly
            # to recur in at least one longitudinal section. Legacy callers
            # without section evidence retain their former behavior.
            if not sections or not session.last_candidate_sections or sections & session.last_candidate_sections:
                session.persistent_candidate_count += 1
            else:
                session.persistent_candidate_count = 1
            session.last_candidate_sections = sections
        else:
            session.persistent_candidate_count = 0
            session.last_candidate_sections.clear()
        # Old callers preserve fail-latching. Hybrid callers explicitly label
        # severe versus provisional evidence.
        severe = (not is_pass) if immediate_failure is None else immediate_failure
        if geometry_score >= 0.55:
            session.geometry_candidate_views += 1
        if "LIKELY_GLARE" in reason_codes:
            session.glare_rejected_views += 1
        if severe:
            session.confirmed_defect_views += 1
        session.has_failure |= severe or session.persistent_candidate_count >= self.weak_candidate_required_views
        if severe or session.persistent_candidate_count >= self.weak_candidate_required_views:
            # Sections run along the cylinder axis, so their screen x-position
            # remains useful even after the defective circumference rotates out
            # of sight. Preserve every confirmed location for the whole part.
            session.confirmed_defect_sections.update(candidate_sections)
        session.worst_score = max(session.worst_score, anomaly_score)
        session.worst_geometry_score = max(session.worst_geometry_score, geometry_score)
        session.worst_tao_score = max(session.worst_tao_score, tao_score if tao_score is not None else anomaly_score)
        session.reason_codes.update(reason_codes)
        completion_triggered = self.completion_mode == "minimum_views" or (
            self.completion_mode == "counting_line" and session.crossed_counting_line
        )
        enough_valid = session.valid_views >= self.minimum_rotation_views
        quality_exhausted = session.frames_inspected >= self.minimum_rotation_views * 2
        if completion_triggered and (enough_valid or quality_exhausted):
            return self._finish(session)
        return None

    def latched_failure(self, track_id: int) -> bool:
        session = self._session_for_track(track_id)
        return bool(session and session.has_failure)

    def defect_sections(self, track_id: int) -> tuple[int, ...]:
        """Return latched longitudinal defect locations for operator guidance."""
        session = self._session_for_track(track_id)
        if session is not None:
            return tuple(sorted(session.confirmed_defect_sections))
        return tuple(sorted(self.completed_defect_sections.get(track_id, ())))

    def needs_initial_inspection(self, track_id: int) -> bool:
        session = self._session_for_track(track_id)
        return session is not None and session.frames_inspected == 0

    def needs_completion_inspection(self, track_id: int) -> bool:
        """Return whether this track crossed and needs its final cropped view."""
        session = self._session_for_track(track_id)
        return bool(
            session
            and (self.completion_mode in {"minimum_views", "part_departure"} or session.crossed_counting_line)
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
            if (self.completion_mode in {"minimum_views", "part_departure"} or session.crossed_counting_line)
            and (session.valid_views >= self.minimum_rotation_views
                 or session.frames_inspected >= self.minimum_rotation_views * 2)
        ]
        self.sessions.clear()
        self.track_to_part.clear()
        self.completed_tracker_ids.clear()
        self.completed_defect_sections.clear()
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
            if session.confirmed_defect_sections:
                self.completed_defect_sections[tracker_id] = frozenset(session.confirmed_defect_sections)
        return completed

    def _session_for_track(self, track_id: int) -> RotatingPartSession | None:
        part_id = self.track_to_part.get(track_id)
        return self.sessions.get(part_id) if part_id is not None else None

    def _entry_side(self, center: float) -> bool:
        forward = self.counting_direction in {"left_to_right", "top_to_bottom"}
        return center < self.counting_line_ratio if forward else center > self.counting_line_ratio

    def _crossed(self, previous: float, current: float) -> bool:
        if self.counting_direction in {"left_to_right", "top_to_bottom"}:
            return previous < self.counting_line_ratio <= current
        return previous > self.counting_line_ratio >= current

    def _center_ratio(self, bounds: tuple[int, int, int, int], frame_width: int,
                      frame_height: int | None) -> float:
        x, y, width, height = bounds
        if self.counting_axis == "y":
            return (y + height / 2.0) / max(frame_height or 1, 1)
        return (x + width / 2.0) / max(frame_width, 1)

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
            reasons.add("INSUFFICIENT_VALID_VIEWS")
            reasons.add("INSUFFICIENT_VIEW_QUALITY")  # legacy storage/API compatibility
        return CompletedPart(session.part_id, "FAIL" if session.has_failure or insufficient else "PASS",
                             session.frames_inspected, session.worst_score, session.valid_views, tuple(sorted(reasons)))
