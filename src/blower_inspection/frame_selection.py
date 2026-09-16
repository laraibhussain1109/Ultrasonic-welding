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
        # A transient bad YOLO box can be a tiny, artificially sharp crop. Do
        # not let it win the burst merely because Laplacian variance increases
        # when the detector zooms into a small textured portion of the blower.
        areas = np.asarray([item.frame.shape[0] * item.frame.shape[1] for item in burst], dtype=np.float32)
        aspects = np.asarray([item.frame.shape[1] / max(item.frame.shape[0], 1) for item in burst])
        median_aspect = float(np.median(aspects))
        eligible = [item for item, area, aspect in zip(burst, areas, aspects)
                    if area >= float(np.max(areas)) * .70
                    and abs(np.log(max(aspect, 1e-6) / max(median_aspect, 1e-6))) <= np.log(1.25)]
        selected = max(eligible or list(burst), key=lambda candidate: candidate.sharpness)
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


class RotationPhaseGate:
    """Accept structurally distinct views rather than repeated stationary frames."""

    def __init__(self, minimum_distance: float = 0.06, maximum_history: int = 24) -> None:
        self.minimum_distance = max(0.0, float(minimum_distance))
        self.maximum_history = max(2, int(maximum_history))
        self._descriptors: dict[int, list[np.ndarray]] = defaultdict(list)

    def accept(self, track_id: int, frame: np.ndarray) -> bool:
        # Local import avoids coupling basic sharp-frame selection to reference
        # bank loading during module import.
        from .reference_bank import structural_descriptor

        descriptor = structural_descriptor(frame)
        history = self._descriptors[track_id]
        if history:
            distance = min(float(np.linalg.norm(descriptor - prior)) for prior in history)
            if distance < self.minimum_distance:
                return False
        history.append(descriptor)
        if len(history) > self.maximum_history:
            del history[0]
        return True

    def discard_missing(self, active_track_ids: set[int]) -> None:
        for track_id in set(self._descriptors) - active_track_ids:
            self._descriptors.pop(track_id, None)

    def clear(self) -> None:
        self._descriptors.clear()
