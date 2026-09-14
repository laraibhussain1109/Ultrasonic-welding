"""Select sharp, temporally separated views of a rotating component."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SelectedFrame:
    frame: np.ndarray
    sharpness: float


class SharpFrameSampler:
    """Collect a short burst and return only its sharpest usable photograph.

    Running inference directly on the video stream lets a single motion-blurred
    frame latch a false failure.  This sampler treats the stream as a sequence
    of photographs: it collects a burst for each tracked part, measures the
    variance of the Laplacian, and submits only the sharpest member.
    """

    def __init__(self, burst_size: int = 5, minimum_sharpness: float = 60.0) -> None:
        if burst_size < 2:
            raise ValueError("burst_size must be at least 2")
        if minimum_sharpness < 0:
            raise ValueError("minimum_sharpness cannot be negative")
        self.burst_size = burst_size
        self.minimum_sharpness = minimum_sharpness
        self._bursts: dict[int, deque[SelectedFrame]] = defaultdict(
            lambda: deque(maxlen=self.burst_size)
        )
        self.rejected_blurry_frames = 0

    @staticmethod
    def sharpness(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def offer(self, track_id: int, frame: np.ndarray) -> SelectedFrame | None:
        burst = self._bursts[track_id]
        burst.append(SelectedFrame(frame.copy(), self.sharpness(frame)))
        if len(burst) < self.burst_size:
            return None
        selected = max(burst, key=lambda candidate: candidate.sharpness)
        self.rejected_blurry_frames += sum(
            candidate.sharpness < self.minimum_sharpness for candidate in burst
        )
        burst.clear()
        return selected if selected.sharpness >= self.minimum_sharpness else None

    def discard_missing(self, active_track_ids: set[int]) -> None:
        for track_id in set(self._bursts) - active_track_ids:
            self._bursts.pop(track_id, None)

    def clear(self) -> None:
        self._bursts.clear()
        self.rejected_blurry_frames = 0
