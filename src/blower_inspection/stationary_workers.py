"""Acquisition and YOLO presence run independently of the UI and PatchCore."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from concurrent.futures import Future

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .camera import crop_bounds
from .capture_performance import EvidenceQueue, StageMeter, CaptureQueueFull
from .config import PartModelConfig
from .frame_quality import FrameQualityAnalyzer
from .stationary_capture import StationaryCapture, StationaryViewCapture, select_stationary_burst


@dataclass(frozen=True)
class CameraSnapshot:
    sequence: int
    captured_at: float
    frame: np.ndarray
    state: str
    fps: float


@dataclass(frozen=True)
class ProcessingFrame:
    frame: np.ndarray
    captured_at: float
    track_id: int
    present: bool
    bounds: tuple[int, int, int, int]


class PendingQuality:
    def __init__(self):
        self.future = Future()

    @property
    def valid(self):
        # Pending evidence is never counted as qualified and never blocks flow.
        return self.future.done() and self.future.result().valid


@dataclass(frozen=True)
class QualityJob:
    frame: np.ndarray
    pending: PendingQuality
    track_id: int


@dataclass(frozen=True)
class BurstJob:
    track_id: int
    angle: int
    burst: tuple
    minimum: int

    @property
    def nbytes(self):
        return sum(frame.nbytes for frame, _quality in self.burst)


@dataclass(frozen=True)
class ResultJob:
    track_id: int
    capture: StationaryCapture

    @property
    def frame(self):
        return self.capture.frame


class DeferredQuality:
    def __init__(self, submit):
        self.submit = submit

    def analyze(self, frame):
        return self.submit(frame)


class StationaryCameraWorker(QThread):
    """Own open/read/release; a separate CPU thread processes ordered evidence.

    The reader does no ROI copy, optical flow, quality, inference or Qt rendering.
    Backpressure never blocks reads: loss of ordered evidence faults inspection.
    """
    failed = pyqtSignal(str)

    def __init__(self, camera, model: PartModelConfig, bounds, track_id: int) -> None:
        super().__init__()
        self.camera, self.model, self.bounds = camera, model, bounds
        self._lock = threading.Lock()
        self._preview_ready = threading.Condition(self._lock)
        self._part = (track_id, True)
        self._latest: CameraSnapshot | None = None
        self._captures = EvidenceQueue(max_items=12)
        self._processing = EvidenceQueue(max_items=64, max_bytes=512 * 1024 * 1024)
        self._quality_jobs = EvidenceQueue(max_items=48)
        self._available = threading.Event()
        self._quality_available = threading.Event()
        self._reader_done = threading.Event()
        self._motion_done = threading.Event()
        self._fault = threading.Event()
        self.error_message = None
        self._state = "STARTING CAMERA"
        self._preview_consumed = 0
        self.preview_overwritten = 0
        self.evidence_dropped = 0
        self.read_meter, self.processing_meter = StageMeter(), StageMeter()
        self.quality_meter, self.motion_meter = StageMeter(), StageMeter()
        self._processing_track = None
        self._processing_pending: dict[int, int] = {}
        self._quality_pending: dict[int, int] = {}

    def set_bounds(self, bounds) -> None:
        with self._lock:
            self.bounds = bounds

    def mark_previewed(self, sequence) -> None:
        with self._lock:
            self._preview_consumed = max(self._preview_consumed, sequence)

    def pending_track_ids(self) -> set[int]:
        with self._lock:
            pending = set(self._processing_pending) | set(self._quality_pending)
            if self._processing_track is not None:
                pending.add(self._processing_track)
        return pending | {track for track, _capture in self._captures}

    def performance(self) -> dict:
        return {"raw": self.read_meter.snapshot(), "processing": self.processing_meter.snapshot(),
                "motion": self.motion_meter.snapshot(), "quality": self.quality_meter.snapshot(),
                "preview_overwritten": self.preview_overwritten, "evidence_dropped": self.evidence_dropped,
                "processing_queue": len(self._processing), "processing_queue_peak": self._processing.high_water,
                "quality_queue": len(self._quality_jobs), "quality_queue_peak": self._quality_jobs.high_water,
                "inspection_queue": len(self._captures), "format": dict(getattr(self.camera, "format_info", {}))}

    def set_part(self, track_id: int, present: bool) -> None:
        with self._lock:
            self._part = (track_id, present)

    def snapshot(self) -> CameraSnapshot | None:
        with self._lock:
            latest, state = self._latest, self._state
        return replace(latest, state=state, fps=self.read_meter.snapshot()["fps"]) if latest else None

    def wait_snapshot(self, after_sequence, timeout=.02):
        """Startup-only wait; steady-state preview polling never blocks."""
        with self._preview_ready:
            self._preview_ready.wait_for(lambda: (self._latest is not None and self._latest.sequence > after_sequence)
                                         or self._fault.is_set(), timeout)
        return self.snapshot()

    def take_captures(self) -> list[tuple[int, StationaryCapture]]:
        with self._lock:
            packets = list(self._captures)
            self._captures.clear()
        return packets

    def _controller(self, track_id) -> StationaryViewCapture:
        model = self.model
        def submit(frame):
            pending = PendingQuality()
            self._submit_quality(QualityJob(frame, pending, track_id))
            return pending
        def select(angle, burst, minimum):
            self._submit_quality(BurstJob(track_id, angle, burst, minimum))
            # Results are published by quality processing, in mechanical order.
            return None
        return StationaryViewCapture(
            burst_frames=model.capture_burst_frames, settle_ms=model.stationary_settle_ms,
            motion_threshold=model.stationary_motion_threshold, flow_threshold=model.stationary_flow_threshold,
            skip_fit_rotation=model.skip_initial_fit_rotation,
            minimum_burst_frames=model.stationary_min_burst_frames,
            burst_window_ms=model.stationary_burst_window_ms,
            quality=DeferredQuality(submit), select_burst=select,
        )

    def _submit_quality(self, job):
        with self._lock:
            self._quality_jobs.append(job)
            self._quality_pending[job.track_id] = self._quality_pending.get(job.track_id, 0) + 1
        self._quality_available.set()

    def _process_quality(self):
        model = self.model
        analyzer = FrameQualityAnalyzer(max(model.minimum_sharpness, model.inference_min_sharpness),
                                        model.max_glare_ratio, model.max_saturation_ratio)
        try:
            while not self._fault.is_set():
                try:
                    job = self._quality_jobs.popleft()
                except IndexError:
                    if self._motion_done.is_set():
                        break
                    self._quality_available.wait(.02)
                    self._quality_available.clear()
                    continue
                if isinstance(job, QualityJob):
                    started = time.monotonic()
                    job.pending.future.set_result(analyzer.analyze(job.frame))
                    self.quality_meter.record(time.monotonic() - started)
                elif isinstance(job, BurstJob):
                    # The FIFO guarantees all this burst's quality jobs finished.
                    burst = [(frame, quality.future.result()) for frame, quality in job.burst]
                    selected = select_stationary_burst(job.angle, burst, job.minimum)
                    with self._lock:
                        self._captures.append((job.track_id, selected))
                else:
                    with self._lock:
                        self._captures.append((job.track_id, job.capture))
                with self._lock:
                    remaining = self._quality_pending[job.track_id] - 1
                    if remaining:
                        self._quality_pending[job.track_id] = remaining
                    else:
                        del self._quality_pending[job.track_id]
        except Exception as exc:
            if isinstance(exc, CaptureQueueFull):
                self.evidence_dropped += 1
            self._fail(f"Stationary quality error: {exc}")

    def _fail(self, message) -> None:
        if not self._fault.is_set():
            self.error_message = message
            self._fault.set()
            self.requestInterruption()
            self.failed.emit(message)

    def _process_frames(self) -> None:
        controller, current = None, None
        try:
            while not self._fault.is_set():
                try:
                    packet = self._processing.popleft()
                except IndexError:
                    if self._reader_done.is_set():
                        break
                    self._available.wait(.02)
                    self._available.clear()
                    continue
                with self._lock:
                    self._processing_track = packet.track_id if packet.present else None
                key = (packet.track_id, packet.bounds)
                if key != current:
                    controller, current = self._controller(packet.track_id), key
                    # Instrument motion without changing the state machine or thresholds.
                    moving = controller._moving
                    def measured_motion(image, moving=moving):
                        started = time.monotonic()
                        result = moving(image)
                        self.motion_meter.record(time.monotonic() - started)
                        return result
                    controller._moving = measured_motion
                started = time.monotonic()
                selected = controller.offer(crop_bounds(packet.frame, packet.bounds, copy=False), now=packet.captured_at) if packet.present else None
                self.processing_meter.record(time.monotonic() - started)
                with self._lock:
                    self._state = controller.state if packet.present else "NO PART"
                    self._processing_track = None
                    if packet.present:
                        remaining = self._processing_pending[packet.track_id] - 1
                        if remaining:
                            self._processing_pending[packet.track_id] = remaining
                        else:
                            del self._processing_pending[packet.track_id]
                if selected is not None:
                    self._submit_quality(ResultJob(packet.track_id, selected))
        except Exception as exc:
            if isinstance(exc, CaptureQueueFull):
                self.evidence_dropped += 1
            self._fail(f"Stationary processing error: {exc}")
        finally:
            self._motion_done.set()
            self._quality_available.set()

    def run(self) -> None:
        sequence = 0
        processor = threading.Thread(target=self._process_frames, name="stationary-motion")
        quality_processor = threading.Thread(target=self._process_quality, name="stationary-native-quality")
        quality_processor.start()
        processor.start()
        try:
            # Windows VideoCapture is created and released on the reader thread.
            if hasattr(self.camera, "open"):
                self.camera.open()
            while not self.isInterruptionRequested():
                # Timestamp before a potentially blocking read, conservatively.
                captured_at = time.monotonic()
                frame = self.camera.read()
                delivered_at = time.monotonic()
                if self.isInterruptionRequested():
                    break
                self.read_meter.record(delivered_at - captured_at, now=delivered_at)
                sequence += 1
                with self._lock:
                    track_id, present = self._part
                    bounds = self.bounds
                    if self._latest is not None and self._latest.sequence > self._preview_consumed:
                        self.preview_overwritten += 1
                    self._latest = CameraSnapshot(sequence, captured_at, frame, self._state, 0.0)
                    self._preview_ready.notify_all()
                    if bounds is not None and self.model.stationary_six_view_capture:
                        self._processing.append(ProcessingFrame(frame, captured_at, track_id, present, bounds))
                        if present:
                            self._processing_pending[track_id] = self._processing_pending.get(track_id, 0) + 1
                self._available.set()
        except Exception as exc:
            if isinstance(exc, CaptureQueueFull):
                self.evidence_dropped += 1
            self._fail(f"Camera acquisition error: {exc}")
        finally:
            try:
                self.camera.close()
            except Exception as exc:
                self._fail(f"Camera release error: {exc}")
            self._reader_done.set()
            self._available.set()
            processor.join()
            quality_processor.join()
            self._processing.clear()
            self._quality_jobs.clear()


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
