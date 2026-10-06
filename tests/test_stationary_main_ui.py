"""Exercise the original UI's six-still queue and firmware verdict boundary."""
import json
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QPushButton

from blower_inspection import app as ui
from blower_inspection.auth import User
from blower_inspection.config import ModelRegistry
from blower_inspection.stationary_capture import INSPECTION_ANGLES, StationaryCapture
from blower_inspection.trainer import InspectionResult
from blower_inspection.yolo_tracking import RotatingPartInspector, TrackedPart


@pytest.fixture
def window(monkeypatch, tmp_path):
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(ui.QSS)
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"active_model": "BF-001", "models": [{
        "id": "BF-001", "name": "Blower", "expected_fins": 24,
        "normal_image_dir": str(tmp_path / "good"), "model_file": str(tmp_path / "model.pt"),
        "result_dir": str(tmp_path / "results"), "runtime_storage_mode": "memory",
        "inspection_completion_mode": "minimum_views", "minimum_rotation_views": 6,
    }]}))
    registry = ModelRegistry(path)
    monkeypatch.setattr(ui, "ModelRegistry", lambda: registry)
    calls = []
    bridge = SimpleNamespace(config=SimpleNamespace(enabled=False), reset=lambda: calls.append("clear"),
                             send_result=lambda failed: calls.append("fail" if failed else "clear"),
                             signal_pass=lambda: calls.append("pass-pulse"), close=lambda: None)
    monkeypatch.setattr(ui, "ESP32FailOutputBridge", lambda: bridge)
    view = ui.InspectionWindow(User("admin", "admin"))
    view.inspection_running = True
    view.rotating_parts = RotatingPartInspector(completion_mode="minimum_views", fixed_view_angles=INSPECTION_ANGLES)
    view.rotating_parts.observe_tracks([TrackedPart(1, (0, 0, 120, 40), .99)], 200)
    yield view, calls
    view.inspection_running = False
    view.inference_worker = None
    view.close()
    application.processEvents()


def test_original_panels_controls_and_tolerance_are_used(window):
    view, _calls = window
    buttons = {button.text() for button in view.findChildren(QPushButton)}
    for label in ("START INSPECTION", "YOLO PART MODEL", "SET PART ROI", "STOP", "MANUAL"):
        assert any(text.endswith(label) for text in buttons)
    assert view.findChild(type(view.viewer), "brand") is not None
    assert not hasattr(view, "tabs")
    minimum_area = view.selected_model().min_defect_area_px
    view.set_tolerance(1)
    assert view.inspection_model().heatmap_tolerance_percent == 1
    view.set_tolerance(8)
    assert view.inspection_model().heatmap_tolerance_percent == 8
    assert view.inspection_model().min_defect_area_px == minimum_area


def test_six_good_stills_emit_one_final_pass_pulse(window):
    view, calls = window
    for angle in INSPECTION_ANGLES:
        result = InspectionResult("PASS", .2, 0, 0, [], view_angle=angle)
        view._handle_inspection_result(1, result, 10)
        if angle != 360:
            assert calls == [] and view.stats["inspected"] == 0
    assert calls == ["pass-pulse"]
    assert view.stats["passed"] == 1
    # The actual dispatcher refuses another inspection of a completed part.
    assert not view.rotating_parts.accepts_inspection(1)


def test_failure_latches_and_entire_section_is_highlighted_after_later_good_stills(window):
    view, calls = window
    for angle in INSPECTION_ANGLES:
        result = InspectionResult("FAIL" if angle == 60 else "PASS", 2, 400, .1, [],
                                  view_angle=angle, candidate_sections=(2,) if angle == 60 else ())
        view._handle_inspection_result(1, result, 10)
    assert "pass-pulse" not in calls and view.stats["failed"] == 1
    assert view.active_fail_asserted and view.rotating_parts.defect_sections(1) == (2,)
    frame = np.zeros((80, 200, 3), np.uint8)
    highlighted = view._draw_tracks(frame, [TrackedPart(1, (0, 20, 120, 40), .99)], .8,
                                    "left_to_right", defect_sections={1: (2,)}, section_count=6)
    assert highlighted[40, 50, 2] > 0
    assert highlighted[40, 50, 0] == 0


def test_slow_inference_preserves_every_queued_angle_in_order(window, monkeypatch):
    view, calls = window
    image = np.full((40, 120, 3), 100, np.uint8)
    workers = []
    class Signal:
        def connect(self, callback):
            self.callback = callback
    class Worker:
        def __init__(self, inspector, model, track_id, frame, view_angle):
            self.track_id, self.angle = track_id, view_angle
            self.finished_result, self.failed, self.finished = Signal(), Signal(), Signal()
            workers.append(self)
        def start(self):
            pass
        def deleteLater(self):
            pass
        def complete(self):
            self.finished_result.callback(self.track_id, InspectionResult("PASS", .1, 0, 0, [], view_angle=self.angle), 50)
            self.finished.callback()
    monkeypatch.setattr(ui, "InspectionWorker", Worker)
    for angle in INSPECTION_ANGLES:
        view.stationary_queue.append((1, StationaryCapture(angle, image, True, 100, burst_frames=15)))
    view._dispatch_stationary_view()
    assert len(view.stationary_queue) == 5
    for _ in range(20):
        view._dispatch_stationary_view()
    assert len(workers) == 1 and len(view.stationary_queue) == 5
    for index in range(6):
        workers[index].complete()
        view._dispatch_stationary_view()
    assert [worker.angle for worker in workers] == list(INSPECTION_ANGLES)
    assert not view.stationary_queue and calls == ["pass-pulse"]


def test_invalid_stop_never_emits_pass_pulse(window):
    view, calls = window
    for angle in INSPECTION_ANGLES:
        view._handle_inspection_result(1, InspectionResult(
            "VIEW INVALID" if angle == 180 else "PASS", 0, 0, 0, [],
            view_valid=angle != 180, view_angle=angle), 0)
    assert "pass-pulse" not in calls and calls == ["fail"]
    assert view.stats["failed"] == 1


def test_result_from_previous_inspection_session_is_discarded(window):
    view, calls = window
    class Signal:
        def connect(self, callback):
            self.callback = callback
    worker = SimpleNamespace(finished_result=Signal(), failed=Signal(), finished=Signal(),
                             start=lambda: None, deleteLater=lambda: None)
    view.inference_worker = worker
    view._start_inference_worker(worker)
    view.inspection_generation += 1
    worker.finished_result.callback(1, InspectionResult("FAIL", 3, 600, .1, [], view_angle=60), 10)
    worker.failed.callback("stale model fault")
    worker.finished.callback()
    assert calls == [] and view.stats["inspected"] == 0
    assert view.rotating_parts.view_progress(1) == (0, 0, 6)
