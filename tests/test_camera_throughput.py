"""Throughput and evidence safety; these are synthetic, not camera certification."""
import os
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from blower_inspection.camera import USBCamera
from blower_inspection.camera_benchmark import benchmark_case
from blower_inspection.capture_performance import CaptureQueueFull, EvidenceQueue, StageMeter, preview_image
from blower_inspection.config import PartModelConfig
from blower_inspection.frame_quality import FrameQualityAnalyzer
from blower_inspection.stationary_capture import INSPECTION_ANGLES, StationaryCapture
from blower_inspection.stationary_workers import ProcessingFrame, StationaryCameraWorker


@pytest.fixture
def app():
    # Keep concurrent motion/quality deterministic under the cloud's four-CPU
    # quota. Native quality and image resolution remain real, not mocked.
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    application = QApplication.instance() or QApplication([])
    yield application
    cv2.setNumThreads(threads)


def settings(tmp_path):
    return PartModelConfig("BF-001", "blower", tmp_path, tmp_path / "model.pt", tmp_path, 24,
                           stationary_settle_ms=200, stationary_min_burst_frames=3,
                           capture_burst_frames=15, stationary_burst_window_ms=350)


class PacedCamera:
    def __init__(self, frames, period=1 / 30):
        self.frames, self.period = iter(frames), period
        self.reads = 0
        self.threads = []
        self.worker = None
        self.closed = False

    def open(self):
        self.threads.append(threading.get_ident())

    def read(self):
        self.threads.append(threading.get_ident())
        time.sleep(self.period)
        frame = next(self.frames, None)
        if frame is None:
            self.worker.requestInterruption()
            return np.zeros((40, 120, 3), np.uint8)
        self.reads += 1
        return frame

    def close(self):
        self.threads.append(threading.get_ident())
        self.closed = True


def test_reader_keeps_running_while_native_quality_and_inference_are_blocked(app, tmp_path, monkeypatch):
    from blower_inspection.app import InspectionWorker
    image = np.random.default_rng(71).integers(40, 180, (80, 240, 3), dtype=np.uint8)
    frames = [np.roll(image, step * 3, 1) for step in range(8)] + [image] * 35
    camera = PacedCamera(frames)
    worker = StationaryCameraWorker(camera, replace(settings(tmp_path), skip_initial_fit_rotation=False), (0, 0, 240, 80), 1)
    camera.worker = worker
    entered, release, inference_entered = threading.Event(), threading.Event(), threading.Event()
    original = FrameQualityAnalyzer.analyze
    def stalled_quality(self, frame, valid_mask=None):
        entered.set()
        release.wait(5)
        return original(self, frame, valid_mask)
    monkeypatch.setattr(FrameQualityAnalyzer, "analyze", stalled_quality)
    class Inspector:
        def inspect(self, *_args, **_kwargs):
            from blower_inspection.trainer import InspectionResult
            inference_entered.set()
            release.wait(5)
            return InspectionResult("PASS", 0, 0, 0, [])
    inference = InspectionWorker(Inspector(), settings(tmp_path), 1, image, 60)
    inference.start()
    worker.start()
    try:
        assert inference_entered.wait(1) and entered.wait(2)
        before = camera.reads
        deadline = time.monotonic() + 1
        while camera.reads < before + 5 and time.monotonic() < deadline:
            time.sleep(.01)
        assert camera.reads >= before + 5
        assert worker.quality_meter.count == 0 and inference.isRunning()
        assert worker.snapshot().sequence == camera.reads
        assert worker.performance()["preview_overwritten"] > 0
    finally:
        release.set()
        assert inference.wait(2000)
        assert worker.wait(4000)
    assert camera.closed and len(set(camera.threads)) == 1
    assert camera.threads[0] != threading.get_ident()
    assert [capture.angle for _track, capture in worker.take_captures()] == [60]


def test_six_one_second_4k_stops_keep_native_roi_and_order_with_200ms_settling(app, tmp_path):
    # Native 4K source, a realistic locked fin ROI; synthetic 30 FPS playback.
    texture = np.random.default_rng(31).integers(40, 180, (360, 640, 3), dtype=np.uint8)
    image = cv2.resize(texture, (3840, 2160), interpolation=cv2.INTER_NEAREST)
    # Moving the full FOV would create a much larger test fixture, so move only
    # the native fin ROI, as the physical fixed camera sees a rotating part.
    bounds = (250, 850, 2800, 440)
    moving = []
    for step in range(8):
        frame = image.copy()
        frame[850:1290, 250:3050] = np.roll(image[850:1290, 250:3050], step * 36, axis=1)
        moving.append(frame)
    def frames():
        for _stop in range(7):  # initial fit/home, then 60..360
            yield from moving
            yield from [image] * 30  # one second at 30 delivered FPS
    camera = PacedCamera(frames())
    worker = StationaryCameraWorker(camera, settings(tmp_path), bounds, 1)
    camera.worker = worker
    worker.start()
    assert worker.wait(18000)
    assert worker.error_message is None, worker.performance()
    captures = worker.take_captures()
    assert [capture.angle for _track, capture in captures] == list(INSPECTION_ANGLES)
    assert all(capture.valid and capture.burst_frames >= 3 for _track, capture in captures)
    for _track, capture in captures:
        assert capture.frame.shape == (440, 2800, 3)
        np.testing.assert_array_equal(capture.frame, image[850:1290, 250:3050])
    assert worker.read_meter.count == camera.reads == 266
    assert worker.evidence_dropped == 0
    assert worker._processing.high_water <= worker._processing.max_items


@pytest.mark.parametrize("byte_limit,item_limit", [(4000, 10), (100000, 1)])
def test_ordered_evidence_overflow_is_explicit_and_does_not_replace_a_view(byte_limit, item_limit):
    image = np.full((40, 30, 3), 100, np.uint8)
    queue = EvidenceQueue(max_items=item_limit, max_bytes=byte_limit)
    queue.append((1, StationaryCapture(60, image, True, 100)))
    with pytest.raises(CaptureQueueFull):
        queue.append((1, StationaryCapture(120, image, True, 100)))
    assert queue.popleft()[1].angle == 60 and not queue


def test_raw_queue_overflow_faults_without_blocking_reader(app, tmp_path, monkeypatch):
    image = np.random.default_rng(2).integers(40, 180, (80, 240, 3), dtype=np.uint8)
    camera = PacedCamera([image] * 100, .01)
    worker = StationaryCameraWorker(camera, settings(tmp_path), (0, 0, 240, 80), 1)
    camera.worker = worker
    worker._processing = EvidenceQueue(max_items=2)
    entered, release = threading.Event(), threading.Event()
    def blocked(_self, _frame, now=None):
        entered.set()
        release.wait(5)
        return None
    monkeypatch.setattr("blower_inspection.stationary_capture.StationaryViewCapture.offer", blocked)
    worker.start()
    try:
        assert entered.wait(1)
        deadline = time.monotonic() + 1
        while worker.error_message is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert "queue full" in worker.error_message
        assert camera.reads == 4 and worker.evidence_dropped == 1
        assert worker.take_captures() == []
    finally:
        release.set()
        assert worker.wait(2000)
    assert camera.closed


def test_benchmark_counts_delivered_fps_and_reports_negotiation_mismatch_and_drop_evidence():
    class Clock:
        value = 0.0
        def __call__(self):
            return self.value
    clock = Clock()
    class Capture:
        sequence = 0
        released = False
        def isOpened(self): return True
        def set(self, *_args): return True
        def getBackendName(self): return "DSHOW"
        def read(self):
            clock.value += .1
            self.sequence += 2
            return True, np.zeros((1080, 1920, 3), np.uint8)
        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"YUY2"),
                    cv2.CAP_PROP_POS_FRAMES: self.sequence}.get(prop, 0)
        def release(self): self.released = True
    capture = Capture()
    result = benchmark_case(0, "DSHOW", 3840, 2160, "MJPG", duration=1, warmup=0,
                            factory=lambda *_args: capture, clock=clock)
    assert result["delivered_fps"] == pytest.approx(10)
    assert result["actual"]["reported_fps"] == 30
    assert not result["resolution_matches_request"] and not result["format_matches_request"]
    assert result["read_latency_ms"]["mean"] == pytest.approx(100)
    drops = result["dropped_frame_evidence"]
    assert drops["driver_sequence_gaps"] > 0
    assert not drops["driver_timestamps_available"]
    assert capture.released


def test_camera_rejects_cross_thread_read_and_release(monkeypatch):
    calls = []
    class Capture:
        def __init__(self, *_args): calls.append(("open", threading.get_ident()))
        def isOpened(self): return True
        def set(self, *_args): return True
        def get(self, _prop): return 0
        def getBackendName(self): return "DSHOW"
        def read(self):
            calls.append(("read", threading.get_ident()))
            return True, np.zeros((40, 120, 3), np.uint8)
        def release(self): calls.append(("release", threading.get_ident()))
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    camera = USBCamera()
    camera.open()
    errors = []
    def unsafe():
        for method in (camera.read, camera.close, camera.open):
            try: method()
            except RuntimeError as exc: errors.append(str(exc))
    thread = threading.Thread(target=unsafe)
    thread.start(); thread.join()
    camera.read(); camera.close()
    assert len(errors) == 3 and len(set(owner for _operation, owner in calls)) == 1
    assert camera.format_info["reported_fps"] is None


def test_preview_is_bounded_before_color_and_qt_conversion_and_source_is_untouched():
    image = np.zeros((2160, 3840, 3), np.uint8)
    preview = preview_image(image, 1100, 620)
    assert preview.shape == (619, 1100, 3)
    preview[:] = 255
    assert image.max() == 0 and image.shape == (2160, 3840, 3)


def test_stage_rates_reflect_completed_reads_and_decay_during_a_stall():
    meter = StageMeter()
    for index in range(31): meter.record(1 / 30, now=index / 30)
    assert meter.snapshot(now=1)["fps"] == pytest.approx(30)
    assert meter.snapshot(now=2)["fps"] == pytest.approx(15)
    assert meter.snapshot(now=1)["mean_ms"] == pytest.approx(1000 / 30)


def test_patchcore_device_policy_selects_cuda_if_available_without_camera_dependency(monkeypatch):
    from blower_inspection.patchcore_inspector import PatchCoreInspector
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                                 device=lambda name: SimpleNamespace(type=name),
                                 backends=SimpleNamespace(cudnn=SimpleNamespace(), cuda=SimpleNamespace(matmul=SimpleNamespace())))
    monkeypatch.delenv("BLOWER_INSPECTION_DEVICE", raising=False)
    assert PatchCoreInspector()._device(fake_torch).type == "cuda"
    fake_torch.cuda.is_available = lambda: False
    assert PatchCoreInspector()._device(fake_torch).type == "cpu"


def test_camera_benchmark_import_does_not_load_torch_or_qt():
    import subprocess, sys
    code = "import sys; import blower_inspection.camera_benchmark; assert 'torch' not in sys.modules; assert 'PyQt6' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_pending_quality_keeps_invalid_later_stop_in_order_and_never_counts_as_pass(app, tmp_path, monkeypatch):
    from blower_inspection.stationary_workers import ResultJob
    image = np.random.default_rng(2).integers(40, 180, (80, 240, 3), dtype=np.uint8)
    worker = StationaryCameraWorker(None, settings(tmp_path), None, 1)
    controller = worker._controller(1)
    entered, release = threading.Event(), threading.Event()
    original = FrameQualityAnalyzer.analyze
    def blocked(analyzer, frame):
        entered.set()
        release.wait(5)
        return original(analyzer, frame)
    monkeypatch.setattr(FrameQualityAnalyzer, "analyze", blocked)
    processor = threading.Thread(target=worker._process_quality)
    processor.start()
    try:
        controller._angle, controller._next_index = 60, 1
        controller._burst = [(image.copy(), controller.quality.analyze(image)) for _ in range(3)]
        assert entered.wait(1)
        assert not any(quality.valid for _frame, quality in controller._burst)
        assert controller._select() is None
        invalid = StationaryCapture(120, image, False, 0, ("NO_SETTLED_STATIONARY_FRAMES",), 0)
        worker._submit_quality(ResultJob(1, invalid))
        assert worker.take_captures() == []
    finally:
        release.set()
        worker._motion_done.set()
        worker._quality_available.set()
        processor.join(3)
    assert not processor.is_alive() and worker.error_message is None
    captures = worker.take_captures()
    assert [(capture.angle, capture.valid) for _track, capture in captures] == [(60, True), (120, False)]


def test_pipeline_audit_saves_all_six_native_stills_and_metrics_without_machine_outputs(app, tmp_path, monkeypatch):
    from blower_inspection import camera_pipeline_benchmark as audit
    image = np.random.default_rng(31).integers(40, 180, (80, 240, 3), dtype=np.uint8)
    moving = [np.roll(image, step * 3, 1) for step in range(8)]
    class Camera(PacedCamera):
        def read(self):
            time.sleep(.01)
            self.reads += 1
            return next(self.frames, image)
    camera = Camera((moving + [image] * 50) * 7)
    monkeypatch.setattr(audit, "USBCamera", lambda *_args: camera)
    monkeypatch.setattr(audit, "ModelRegistry", lambda: SimpleNamespace(get=lambda _id: settings(tmp_path)))
    destination = tmp_path / "audit"
    result = audit.main(["--roi", "0", "0", "240", "80", "--width", "240", "--height", "80", "--seconds", "10", "--output", str(destination)])
    import json
    report = json.loads((destination / "report.json").read_text())
    assert result == 0 and report["six_valid_views"] and report["inference_complete"]
    assert [capture["angle"] for capture in report["captures"]] == list(INSPECTION_ANGLES)
    assert report["final"]["evidence_dropped"] == 0 and report["samples"]
    for angle in INSPECTION_ANGLES:
        np.testing.assert_array_equal(cv2.imread(str(destination / f"{angle:03d}.png")), image)
    assert camera.closed and report["inference_device"] == "inference disabled"
