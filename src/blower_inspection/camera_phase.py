"""Camera-only indexing for stations where the PLC exposes no phase output.

The sequencer never identifies a phase by matching surface appearance.  It only
detects moving/stopped transitions and assigns the known mechanical stop order
after observing the initial continuous fitment revolution return to HOME.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .phase_locked import ANGLES


@dataclass(frozen=True)
class CameraPhaseUpdate:
    state: str
    angle: int | None
    burst_count: int
    burst: tuple[np.ndarray, ...] = ()
    home_frame: np.ndarray | None = None
    motion_score: float = 0.0


class CameraPhaseSequencer:
    """Infer motor motion boundaries, then count deterministic indexed stops."""

    def __init__(self, *, angles: tuple[int, ...] = ANGLES, settle_delay_ms: int = 200,
                 burst_frame_count: int = 7, motion_threshold: float = 6.0,
                 moving_confirmation_frames: int = 2, stationary_confirmation_frames: int = 2) -> None:
        self.angles = angles
        self.settle_delay_ms = max(0, settle_delay_ms)
        self.burst_frame_count = max(1, burst_frame_count)
        self.motion_threshold = max(.1, motion_threshold)
        self.moving_confirmation_frames = max(1, moving_confirmation_frames)
        self.stationary_confirmation_frames = max(1, stationary_confirmation_frames)
        self.reset()

    def reset(self) -> None:
        self.state = "WAITING FOR FITMENT ROTATION"
        self._previous: np.ndarray | None = None
        self._moving_count = 0
        self._stationary_count = 0
        self._observed_motion = False
        self._phase_index = 0
        self._active_angle: int | None = None
        self._stopped_at = 0.0
        self._burst: list[np.ndarray] = []
        self.home_frame: np.ndarray | None = None
        self.motion_score = 0.0

    @staticmethod
    def _motion_image(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        height = 96
        width = max(160, int(gray.shape[1] * height / max(gray.shape[0], 1)))
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        # Suppress sensor noise and fine specular shimmer while retaining the
        # coherent fin displacement caused by actual motor rotation.
        return cv2.GaussianBlur(resized, (5, 5), 0)

    def offer(self, frame: np.ndarray, now: float | None = None) -> CameraPhaseUpdate:
        now = time.monotonic() if now is None else now
        current = self._motion_image(frame)
        if self._previous is None:
            self._previous = current
            return self._update()
        score = float(np.mean(cv2.absdiff(current, self._previous)))
        self.motion_score = score
        self._previous = current
        # Hysteresis prevents a stopped reflective part from remaining forever
        # in FITMENT ROTATION because its camera noise sits near one threshold.
        threshold = self.motion_threshold if not self._observed_motion else self.motion_threshold * .55
        moving = score >= threshold
        if moving:
            self._moving_count += 1
            self._stationary_count = 0
            self._burst.clear()
            if self._moving_count >= self.moving_confirmation_frames:
                self._observed_motion = True
                self._active_angle = None
                self.state = "FITMENT ROTATION" if self.home_frame is None else "ROTATING TO NEXT POSITION"
            return self._update()

        self._stationary_count += 1
        self._moving_count = 0
        if not self._observed_motion or self._stationary_count < self.stationary_confirmation_frames:
            return self._update()

        # First confirmed stop after observed continuous motion is HOME. Later
        # confirmed stops are numbered solely by deterministic mechanical order.
        if self.home_frame is None:
            self.home_frame = frame.copy()
            self._observed_motion = False
            self._stopped_at = now
            self.state = "HOME LOCKED — WAITING FOR 60°"
            return self._update(home_frame=self.home_frame)
        if self._active_angle is None:
            if self._phase_index >= len(self.angles):
                self.state = "INDEXED INSPECTION COMPLETE"
                return self._update()
            self._active_angle = self.angles[self._phase_index]
            self._stopped_at = now
            self._observed_motion = False
            self.state = f"SETTLING {self._active_angle}°"
        if (now - self._stopped_at) * 1000 < self.settle_delay_ms:
            return self._update()
        self.state = f"CAPTURING {self._active_angle}°"
        self._burst.append(frame.copy())
        if len(self._burst) < self.burst_frame_count:
            return self._update()
        burst = tuple(self._burst)
        angle = self._active_angle
        self._burst = []
        self._phase_index += 1
        self._active_angle = None
        self.state = "WAITING FOR NEXT ROTATION" if self._phase_index < len(self.angles) else "INDEXED INSPECTION COMPLETE"
        return CameraPhaseUpdate(self.state, angle, len(burst), burst, self.home_frame, self.motion_score)

    def _update(self, *, home_frame: np.ndarray | None = None) -> CameraPhaseUpdate:
        return CameraPhaseUpdate(self.state, self._active_angle, len(self._burst), (), home_frame, self.motion_score)
