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
    pixel_motion: float = 0.0
    flow_motion: float = 0.0


class CameraPhaseSequencer:
    """Infer motor motion boundaries, then count deterministic indexed stops."""

    def __init__(self, *, angles: tuple[int, ...] = ANGLES, settle_delay_ms: int = 200,
                 burst_frame_count: int = 7, motion_threshold: float = 2.5,
                 motion_flow_threshold: float = .35,
                 motion_pixel_guard_ratio: float = .35,
                 moving_confirmation_frames: int = 2, stationary_confirmation_frames: int = 3,
                 lock_initial_home: bool = True) -> None:
        self.angles = angles
        self.settle_delay_ms = max(0, settle_delay_ms)
        self.burst_frame_count = max(1, burst_frame_count)
        self.motion_threshold = max(.1, motion_threshold)
        self.motion_flow_threshold = max(.01, motion_flow_threshold)
        self.motion_pixel_guard_ratio = min(1.0, max(.05, motion_pixel_guard_ratio))
        self.moving_confirmation_frames = max(1, moving_confirmation_frames)
        self.stationary_confirmation_frames = max(1, stationary_confirmation_frames)
        self.lock_initial_home = lock_initial_home
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
        self.pixel_motion = 0.0
        self.flow_motion = 0.0

    @staticmethod
    def _motion_image(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        height = 96
        width = max(160, int(gray.shape[1] * height / max(gray.shape[0], 1)))
        reduced = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        # Suppress sensor noise and tiny exposure shimmer before optical flow;
        # these are not mechanical rotation.
        return cv2.GaussianBlur(reduced, (5, 5), 0)

    def offer(self, frame: np.ndarray, now: float | None = None) -> CameraPhaseUpdate:
        now = time.monotonic() if now is None else now
        current = self._motion_image(frame)
        if self._previous is None:
            self._previous = current
            # Camera-only stations have no PLC event with which to distinguish
            # the fitment revolution from the indexed revolution.  Preserve the
            # stationary image visible when inspection is armed as HOME, then
            # treat every subsequent motion -> stop transition as 60..360.
            # This avoids consuming the first real indexed stop merely to learn
            # HOME (the cause of the former one-phase/no-phase behaviour).
            if self.lock_initial_home:
                self.home_frame = frame.copy()
                self.state = "HOME LOCKED — WAITING FOR 60°"
            return self._update(home_frame=self.home_frame)
        score = float(np.mean(cv2.absdiff(current, self._previous)))
        flow = cv2.calcOpticalFlowFarneback(
            self._previous, current, None, .5, 3, 15, 2, 5, 1.2, 0
        )
        flow_score = float(np.median(cv2.magnitude(flow[..., 0], flow[..., 1])))
        self.pixel_motion, self.flow_motion = score, flow_score
        self._previous = current
        # Repetitive fins can translate by one pitch and still have a deceptively
        # small pixel difference. Optical-flow magnitude closes that aliasing
        # hole; either independent motion signal prevents burst capture.
        # Farneback can report 0.4–0.8 px of apparent flow on a stationary,
        # reflective part from camera noise alone. Flow therefore needs a small
        # independent pixel-change guard. Large raw change remains sufficient,
        # while pitch-aliased fin motion is caught by flow plus modest change.
        moving = (
            score >= self.motion_threshold
            or (flow_score >= self.motion_flow_threshold
                and score >= self.motion_threshold * self.motion_pixel_guard_ratio)
        )
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
        return CameraPhaseUpdate(self.state, angle, len(burst), burst, self.home_frame,
                                 self.pixel_motion, self.flow_motion)

    def _update(self, *, home_frame: np.ndarray | None = None) -> CameraPhaseUpdate:
        return CameraPhaseUpdate(self.state, self._active_angle, len(self._burst), (), home_frame,
                                 self.pixel_motion, self.flow_motion)

    def discard_partial_burst(self) -> None:
        """Drop buffered transition frames without changing the indexed phase."""
        self._burst.clear()
