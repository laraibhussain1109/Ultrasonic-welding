"""Real acquisition threads retain every stop when UI/YOLO processing is slow."""
import os
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from blower_inspection.config import PartModelConfig
from blower_inspection.stationary_capture import INSPECTION_ANGLES
from blower_inspection.stationary_workers import StationaryCameraWorker, YoloPresenceWorker


def model(tmp_path):
    return PartModelConfig("BF-001", "blower", tmp_path, tmp_path / "model.pt", tmp_path, 24,
                           capture_burst_frames=15, stationary_settle_ms=0,
                           stationary_burst_window_ms=350)


class VideoCamera:
    def __init__(self, image, frames):
        self.image, self.frames = image, iter(frames)
        self.reads, self.closed = 0, False
        self.worker = None
    def read(self):
        time.sleep(.01)
        frame = next(self.frames, None)
        if frame is None:
            self.worker.requestInterruption()
            return self.image
        self.reads += 1
        return frame
    def close(self):
        self.closed = True


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_all_six_short_stops_are_captured_while_presence_worker_is_blocked(app, tmp_path):
    image = np.random.default_rng(72).integers(40, 180, (80, 240, 3), dtype=np.uint8)
    frames = []
    for index in range(7):  # fit rotation, then six indexed stops
        frames.extend(np.roll(image, step * 3, axis=1) for step in range(8))
        frames.extend([image] * (10 if index < 6 else 30))
    camera = VideoCamera(image, frames)
    worker = StationaryCameraWorker(camera, model(tmp_path), (0, 0, 240, 80), 1)
    camera.worker = worker
    release, entered = threading.Event(), threading.Event()
    class Detector:
        def detect_best(self, frame):
            entered.set()
            release.wait(10)
            return SimpleNamespace(confidence=.99)
    presence = YoloPresenceWorker(Detector(), image, .9)
    errors = []
    worker.failed.connect(errors.append)
    presence.start()
    assert entered.wait(1)
    worker.start()
    try:
        # No UI frame polling or event processing while camera acquisition runs.
        assert worker.wait(8000)
        assert presence.isRunning()
        packets = worker.take_captures()
        assert [packet.angle for _track, packet in packets] == list(INSPECTION_ANGLES)
        assert all(packet.valid for _track, packet in packets)
        assert all(3 <= packet.burst_frames < 15 for _track, packet in packets[:-1])
        assert worker.snapshot().sequence == len(frames) == camera.reads
        assert worker.take_captures() == [] and camera.closed
        app.processEvents()
        assert errors == []
    finally:
        worker.requestInterruption()
        worker.wait(2000)
        release.set()
        assert presence.wait(2000)
        app.processEvents()


def test_stopping_acquisition_releases_camera_and_no_late_frame_is_published(app, tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Camera:
        closed = False
        def read(self):
            entered.set()
            release.wait(5)
            return np.full((40, 120, 3), 100, np.uint8)
        def close(self):
            self.closed = True
    camera = Camera()
    worker = StationaryCameraWorker(camera, model(tmp_path), (0, 0, 120, 40), 1)
    worker.start()
    assert entered.wait(1)
    worker.requestInterruption()
    release.set()
    assert worker.wait(2000)
    assert camera.closed and worker.snapshot() is None and worker.take_captures() == []


def test_new_part_id_resets_phases_without_reusing_previous_part_capture(app, tmp_path):
    image = np.random.default_rng(31).integers(40, 180, (80, 240, 3), dtype=np.uint8)
    frames = [np.roll(image, step * 3, axis=1) for step in range(8)] + [image] * 30
    camera = VideoCamera(image, frames)
    settings = replace(model(tmp_path), skip_initial_fit_rotation=False)
    worker = StationaryCameraWorker(camera, settings, (0, 0, 240, 80), 2)
    camera.worker = worker
    worker.set_part(3, True)
    worker.start()
    assert worker.wait(5000)
    packets = worker.take_captures()
    assert len(packets) == 1 and packets[0][0] == 3 and packets[0][1].angle == 60
