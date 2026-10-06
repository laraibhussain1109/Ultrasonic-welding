"""Select one photograph per mechanical 60-degree stop from a camera stream.

Angles follow the station's configured stop order, never surface similarity.
Camera motion is evidence of moving/stopped transitions, not an encoder.
"""
from __future__ import annotations

import time
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


class StationaryViewCapture:
    def __init__(self, *, burst_frames: int = 15, settle_ms: int = 200,
                 motion_threshold: float = 2.5, flow_threshold: float = .35,
                 skip_fit_rotation: bool = True,
                 quality: FrameQualityAnalyzer | None = None) -> None:
        self.burst_frames = min(30, max(10, burst_frames))
        self.settle_s = max(0, settle_ms) / 1000
        self.motion_threshold = motion_threshold
        self.flow_threshold = flow_threshold
        self.skip_fit_rotation = skip_fit_rotation
        self.quality = quality or FrameQualityAnalyzer()
        self.state = "WAITING FOR FIT ROTATION" if skip_fit_rotation else "WAITING FOR ROTATION"
        self._home = not skip_fit_rotation
        self._previous = None
        self._moving_frames = 0
        self._stopped_frames = 0
        self._motion_seen = False
        self._next_index = 0
        self._angle: int | None = None
        self._stopped_at = 0.0
        self._burst: list[tuple[np.ndarray, FrameQuality]] = []

    @staticmethod
    def _motion_image(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        # Use the fin band, excluding stationary background around the fixture.
        h, w = gray.shape
        gray = gray[round(h * .18):max(round(h * .82), round(h * .18) + 1)]
        return cv2.resize(gray, (min(320, max(80, w)), 64), interpolation=cv2.INTER_AREA)

    def _moving(self, current: np.ndarray) -> bool:
        if self._previous is None or current.shape != self._previous.shape:
            self._previous = current
            return False
        before = self._previous
        self._previous = current
        # Global exposure drift alone must not invent a new stop.
        delta = current.astype(np.float32) - before.astype(np.float32)
        difference = float(np.mean(np.abs(delta - np.median(delta))))
        flow = cv2.calcOpticalFlowFarneback(before, current, None, .5, 3, 15, 2, 5, 1.2, 0)
        movement = float(np.quantile(cv2.magnitude(flow[..., 0], flow[..., 1]), .80))
        return difference >= self.motion_threshold or movement >= self.flow_threshold

    def _select(self, *, incomplete: bool = False) -> StationaryCapture:
        assert self._angle is not None and self._burst
        eligible = [(frame, quality) for frame, quality in self._burst if quality.valid]
        frame, quality = max(eligible or self._burst, key=lambda item: item[1].sharpness)
        reasons = (("INCOMPLETE_STOP_BURST",) if incomplete else quality.reasons)
        result = StationaryCapture(self._angle, frame.copy(), bool(eligible) and not incomplete,
                                   quality.sharpness, reasons, len(self._burst))
        self._burst.clear()
        self._angle = None
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
                if not self._burst:
                    self._burst.append((frame.copy(), self.quality.analyze(frame)))
                interrupted = self._select(incomplete=True)
            self._motion_seen = True
            self.state = "FIT ROTATION" if not self._home else "ROTATING"
            return interrupted
        self._moving_frames = 0
        self._stopped_frames += 1
        if self._stopped_frames < 3:
            return None
        if self._motion_seen:
            self._motion_seen = False
            self._stopped_at = now
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
        self._burst.append((frame.copy(), self.quality.analyze(frame)))
        if len(self._burst) >= self.burst_frames:
            return self._select()
        return None
