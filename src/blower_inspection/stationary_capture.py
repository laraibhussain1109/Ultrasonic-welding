"""Select one photograph per mechanical 60-degree stop from a camera stream.

Angles follow the station's configured stop order, never surface similarity.
Camera motion is evidence of moving/stopped transitions, not an encoder.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .frame_quality import FrameQuality, FrameQualityAnalyzer

INSPECTION_ANGLES = (60, 120, 180, 240, 300, 360)


@dataclass(frozen=True)
class StationaryCapture:
    angle: int
    frame: np.ndarray
    valid: bool
    sharpness: float
    reasons: tuple[str, ...] = ()
    burst_frames: int = 0


def select_stationary_burst(angle, burst, minimum_burst_frames) -> StationaryCapture:
    """Shared selection rule for synchronous and deferred native quality checks."""
    eligible = [(frame, quality) for frame, quality in burst if quality.valid]
    frame, quality = max(eligible or burst, key=lambda item: item[1].sharpness)
    enough = len(eligible) >= minimum_burst_frames
    reasons = (() if enough else tuple(dict.fromkeys(("INSUFFICIENT_STATIONARY_FRAMES",) + quality.reasons)))
    return StationaryCapture(angle, frame.copy(), enough, quality.sharpness, reasons, len(burst))


class StationaryViewCapture:
    def __init__(self, *, burst_frames: int = 15, settle_ms: int = 200,
                 motion_threshold: float = 2.5, flow_threshold: float = .35,
                 skip_fit_rotation: bool = True,
                 minimum_burst_frames: int = 3, burst_window_ms: int = 350,
                 quality: FrameQualityAnalyzer | None = None, select_burst=None) -> None:
        self.burst_frames = min(30, max(10, burst_frames))
        self.minimum_burst_frames = min(self.burst_frames, max(2, minimum_burst_frames))
        self.burst_window_s = max(0, burst_window_ms) / 1000
        self.settle_s = max(0, settle_ms) / 1000
        self.motion_threshold = motion_threshold
        self.flow_threshold = flow_threshold
        self.skip_fit_rotation = skip_fit_rotation
        self.quality = quality or FrameQualityAnalyzer()
        self.select_burst = select_burst or select_stationary_burst
        self.state = "WAITING FOR FIT ROTATION" if skip_fit_rotation else "WAITING FOR ROTATION"
        self._home = not skip_fit_rotation
        self._previous = None
        self._motion_history = deque(maxlen=4)
        self._moving_frames = 0
        self._stopped_frames = 0
        self._motion_seen = False
        self._next_index = 0
        self._angle: int | None = None
        self._stopped_at = 0.0
        self._burst_started_at = 0.0
        self._burst: list[tuple[np.ndarray, FrameQuality]] = []

    @staticmethod
    def _motion_image(frame: np.ndarray) -> np.ndarray:
        # Use the fin band, excluding stationary background around the fixture.
        h, w = frame.shape[:2]
        band = frame[round(h * .18):max(round(h * .82), round(h * .18) + 1)]
        reduced = cv2.resize(band, (min(320, max(80, w)), 64), interpolation=cv2.INTER_AREA)
        if reduced.ndim == 3:
            reduced = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(reduced, (3, 3), 0)

    def _moving(self, current: np.ndarray) -> bool:
        if self._previous is None or current.shape != self._previous.shape:
            self._previous = current
            self._motion_history.clear()
            self._motion_history.append(current)
            return False
        before = self._previous
        self._previous = current
        # Global exposure drift alone must not invent a new stop.
        delta = current.astype(np.float32) - before.astype(np.float32)
        difference = float(np.mean(np.abs(delta - np.median(delta))))
        normalized = np.clip(current.astype(np.float32) - np.median(delta), 0, 255).astype(np.uint8)
        anchor = self._motion_history[0]
        self._motion_history.append(current)
        if np.array_equal(before, normalized):
            # Exact stills/uniform brightness drift need no optical flow. Keep
            # the temporal baseline until a subtle step has fully stopped.
            return self._localized_motion(anchor, current) if not np.array_equal(anchor, current) else False
        flow = cv2.calcOpticalFlowFarneback(before, normalized, None, .5, 3, 15, 2, 5, 1.2, 0)
        movement = float(np.quantile(cv2.magnitude(flow[..., 0], flow[..., 1]), .80))
        # A blower may change only a small portion of its visible fin band. The
        # old global 80th percentile misses that evidence. Follow corner tracks
        # within individual axial sections, including a short temporal baseline
        # so slow subpixel steps accumulate rather than being discarded.
        return (difference >= self.motion_threshold or movement >= self.flow_threshold
                or self._localized_motion(anchor, current))

    def _localized_motion(self, before, current):
        offset = np.median(current.astype(np.float32) - before.astype(np.float32))
        normalized = np.clip(current.astype(np.float32) - offset, 0, 255).astype(np.uint8)
        if np.array_equal(before, normalized):
            return False
        points = cv2.goodFeaturesToTrack(before, maxCorners=120, qualityLevel=.05, minDistance=4)
        if points is None or len(points) < 4:
            return False
        following, status, error = cv2.calcOpticalFlowPyrLK(before, normalized, points, None,
                                                          winSize=(15, 15), maxLevel=2)
        if following is None:
            return False
        # Forward/backward agreement rejects disappearing highlights and bad
        # matches. Brightness-only changes were removed before calculating flow.
        back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(normalized, before, following, None,
                                                        winSize=(15, 15), maxLevel=2)
        if back is None:
            return False
        keep = ((status.ravel() == 1) & (reverse_status.ravel() == 1)
                & (error.ravel() < 12) & (np.linalg.norm((back - points).reshape(-1, 2), axis=1) < .5))
        coordinates = points.reshape(-1, 2)[keep]
        vectors = (following - points).reshape(-1, 2)[keep]
        for section in range(6):
            local = vectors[(coordinates[:, 0] >= section * before.shape[1] / 6)
                            & (coordinates[:, 0] < (section + 1) * before.shape[1] / 6)]
            if len(local) < 4:
                continue
            median = np.median(local, axis=0)
            distance = float(np.linalg.norm(median))
            coherent = np.mean(np.linalg.norm(local - median, axis=1) <= max(.15, distance * .5))
            if distance >= self.flow_threshold and coherent >= .75:
                return True
        return False

    def _select(self) -> StationaryCapture:
        assert self._angle is not None and self._burst
        result = self.select_burst(self._angle, tuple(self._burst), self.minimum_burst_frames)
        self._burst.clear()
        self._angle = None
        self._burst_started_at = 0.0
        self.state = "SIX STOPS CAPTURED" if self._next_index == 6 else "WAITING FOR ROTATION"
        return result

    def offer(self, frame: np.ndarray, now: float | None = None) -> StationaryCapture | None:
        now = time.monotonic() if now is None else now
        if self._moving(self._motion_image(frame)):
            self._moving_frames += 1
            self._stopped_frames = 0
            self._stopped_at = now
            if self._moving_frames < 2:
                return None
            interrupted = None
            if self._angle is not None:
                if self._burst:
                    # Only stopped, settled frames already buffered are eligible.
                    # The current moving frame must never become the selected still.
                    interrupted = self._select()
                else:
                    interrupted = StationaryCapture(self._angle, frame.copy(), False, 0.0,
                                                   ("NO_SETTLED_STATIONARY_FRAMES",), 0)
                    self._angle = None
            self._motion_seen = True
            self.state = "FIT ROTATION" if not self._home else "ROTATING"
            return interrupted
        self._moving_frames = 0
        self._stopped_frames += 1
        if self._stopped_frames == 1:
            self._stopped_at = now
        if self._stopped_frames < 3:
            return None
        if self._motion_seen:
            self._motion_seen = False
            if not self._home:
                self._home = True
                self.state = "HOME — WAITING FOR 60°"
                return None
            if self._next_index < len(INSPECTION_ANGLES):
                self._angle = INSPECTION_ANGLES[self._next_index]
                self._next_index += 1
                self.state = f"SETTLING {self._angle}°"
        if self._angle is None or now - self._stopped_at < self.settle_s:
            return None
        self.state = f"CAPTURING {self._angle}°"
        if not self._burst:
            self._burst_started_at = now
        candidate = frame.copy()
        self._burst.append((candidate, self.quality.analyze(candidate)))
        if len(self._burst) >= self.burst_frames or (
            now - self._burst_started_at >= self.burst_window_s
            and sum(quality.valid for _frame, quality in self._burst) >= self.minimum_burst_frames
        ):
            return self._select()
        return None
