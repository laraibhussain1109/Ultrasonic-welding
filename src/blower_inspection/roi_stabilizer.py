"""Canonical, aspect-preserving blower ROI shared by training and inference."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CanonicalROI:
    image: np.ndarray
    bounds: tuple[int, int, int, int]
    content_bounds: tuple[int, int, int, int]


class ROIStabilizer:
    def __init__(self, output_size: tuple[int, int] = (640, 256), *, mode: str = "yolo_stabilized",
                 smoothing_frames: int = 5, padding_ratio: float = .04,
                 fixed_bounds: tuple[int, int, int, int] | None = None) -> None:
        if mode not in {"yolo_stabilized", "fixed_after_detection"}:
            raise ValueError("roi_mode must be yolo_stabilized or fixed_after_detection")
        self.output_size, self.mode = output_size, mode
        self.smoothing_frames = max(1, int(smoothing_frames))
        self.padding_ratio, self.fixed_bounds = max(0.0, float(padding_ratio)), fixed_bounds
        self._history: dict[int, deque[tuple[int, int, int, int]]] = defaultdict(
            lambda: deque(maxlen=self.smoothing_frames))

    def stabilize(self, bounds: tuple[int, int, int, int], shape: tuple[int, ...], track_id: int = 0) -> tuple[int, int, int, int]:
        if self.mode == "fixed_after_detection":
            if self.fixed_bounds is None:
                self.fixed_bounds = bounds
            selected = self.fixed_bounds
        else:
            history = self._history[track_id]
            history.append(bounds)
            selected = tuple(int(round(v)) for v in np.median(np.asarray(history), axis=0))
        x, y, w, h = selected
        pad_x, pad_y = round(w * self.padding_ratio), round(h * self.padding_ratio)
        height, width = shape[:2]
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def canonicalize(self, frame: np.ndarray, bounds: tuple[int, int, int, int], track_id: int = 0) -> CanonicalROI:
        stable = self.stabilize(bounds, frame.shape, track_id)
        x, y, w, h = stable
        crop = frame[y:y + h, x:x + w]
        out_w, out_h = self.output_size
        scale = min(out_w / max(w, 1), out_h / max(h, 1))
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((out_h, out_w) + (() if frame.ndim == 2 else (frame.shape[2],)), dtype=frame.dtype)
        ox, oy = (out_w - new_w) // 2, (out_h - new_h) // 2
        canvas[oy:oy + new_h, ox:ox + new_w] = resized
        return CanonicalROI(canvas, stable, (ox, oy, new_w, new_h))

    def clear(self) -> None:
        self._history.clear()

