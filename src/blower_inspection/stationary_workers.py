"""Acquisition and YOLO presence run independently of the UI and PatchCore."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .camera import crop_bounds
from .config import PartModelConfig
from .frame_quality import FrameQualityAnalyzer
from .stationary_capture import StationaryCapture, StationaryViewCapture


@dataclass(frozen=True)
class CameraSnapshot:
    sequence: int
    captured_at: float
    frame: np.ndarray
    state: str
    fps: float


class StationaryCameraWorker(QThread):
    """Drain every camera frame; retain all stop results but only the latest preview."""
    failed = pyqtSignal(str)

    def __init__(self, camera, model: PartModelConfig, bounds, track_id: int) -> None:
        super().__init__()
        self.camera, self.model, self.bounds = camera, model, bounds
        self._lock = threading.Lock()
        self._part = (track_id, True)
        self._latest: CameraSnapshot | None = None
        self._captures: deque[tuple[int, StationaryCapture]] = deque()

    def set_part(self, track_id: int, present: bool) -> None:
        with self._lock:
            self._part = (track_id, present)

    def snapshot(self) -> CameraSnapshot | None:
        with self._lock:
            return self._latest

    def take_captures(self) -> list[tuple[int, StationaryCapture]]:
        with self._lock:
            packets = list(self._captures)
            self._captures.clear()
        return packets

    def _controller(self) -> StationaryViewCapture:
        model = self.model
        return StationaryViewCapture(
            burst_frames=model.capture_burst_frames, settle_ms=model.stationary_settle_ms,
            motion_threshold=model.stationary_motion_threshold, flow_threshold=model.stationary_flow_threshold,
            skip_fit_rotation=model.skip_initial_fit_rotation,
            minimum_burst_frames=model.stationary_min_burst_frames,
            burst_window_ms=model.stationary_burst_window_ms,
            quality=FrameQualityAnalyzer(max(model.minimum_sharpness, model.inference_min_sharpness),
                                       model.max_glare_ratio, model.max_saturation_ratio),
        )

    def run(self) -> None:
        controller = self._controller()
        current_track = self._part[0]
        sequence = 0
        started = time.monotonic()
        try:
            while not self.isInterruptionRequested():
                # Timestamp before a potentially blocking read, conservatively.
                captured_at = time.monotonic()
                frame = self.camera.read()
                if self.isInterruptionRequested():
                    break
                with self._lock:
                    track_id, present = self._part
                if track_id != current_track:
                    current_track, controller = track_id, self._controller()
                selected = controller.offer(crop_bounds(frame, self.bounds), now=captured_at) if present else None
                sequence += 1
                snapshot = CameraSnapshot(sequence, captured_at, frame,
                                          controller.state if present else "NO PART",
                                          sequence / max(time.monotonic() - started, .001))
                with self._lock:
                    self._latest = snapshot
                    if selected is not None:
                        if len(self._captures) >= 64:
                            raise RuntimeError("Stationary image queue full; stop the machine and restart inspection")
                        self._captures.append((track_id, selected))
        except Exception as exc:
            self.failed.emit(f"Camera acquisition error: {exc}")
        finally:
            self.camera.close()


class YoloPresenceWorker(QThread):
    result = pyqtSignal(bool, float)
    failed = pyqtSignal(str)

    def __init__(self, detector, frame: np.ndarray, confidence: float) -> None:
        super().__init__()
        self.detector, self.frame, self.confidence = detector, frame.copy(), confidence

    def run(self) -> None:
        try:
            detection = self.detector.detect_best(self.frame)
            self.result.emit(detection.confidence >= self.confidence, time.monotonic())
        except ValueError:
            self.result.emit(False, time.monotonic())
        except Exception as exc:
            self.failed.emit(f"YOLO presence check failed: {exc}")
