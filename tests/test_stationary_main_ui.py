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


def test_video_inference_does_not_wait_for_a_rotation_event_or_emit_pass(window):
    view, calls = window
    for _frame in range(20):
        view._handle_inspection_result(1, InspectionResult("PASS", .2, 0, 0, []), 15, video_evidence=True)
    assert calls == [] and view.stats["inspected"] == 0
    assert view.rotating_parts.view_progress(1) == (0, 0, 6)
    assert "COVERAGE UNCONFIRMED" in view.last_result.text()
    view._handle_inspection_result(1, InspectionResult("FAIL", 2, 300, .1, [],
                                  candidate_sections=(3,)), 15, video_evidence=True)
    assert calls == ["fail"] and view.rotating_parts.defect_sections(1) == (3,)
    for angle in INSPECTION_ANGLES:
        view._handle_inspection_result(1, InspectionResult("PASS", .1, 0, 0, [], view_angle=angle), 15)
    assert "pass-pulse" not in calls and view.stats["failed"] == 1


def test_video_dispatch_is_serial_and_required_stills_have_priority(window, monkeypatch):
    from blower_inspection.stationary_workers import VideoEvidence
    view, calls = window
    image = np.random.default_rng(24).integers(40, 180, (40, 120, 3), dtype=np.uint8)
    consumed, started = [], []
    view.camera_worker = SimpleNamespace(take_video_evidence=lambda: consumed.append(1) or VideoEvidence(1, image, 0))
    monkeypatch.setattr(view, "_start_inference_worker", lambda worker: started.append(worker))
    view._dispatch_stationary_view()
    assert len(started) == 1 and started[0].video_evidence and started[0].view_angle is None
    view.stationary_queue.append((1, StationaryCapture(60, image, True, 100, burst_frames=3)))
    view._dispatch_stationary_view()
    assert len(started) == 1 and len(consumed) == 1 and len(view.stationary_queue) == 1
    view.inference_worker = None
    view._dispatch_stationary_view()
    assert len(started) == 2 and started[1].view_angle == 60 and not started[1].video_evidence
    assert len(consumed) == 1 and not calls
    view.camera_worker = None


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


def test_fast_acquisition_snapshot_replaces_blocking_gui_camera_read(window, monkeypatch):
    import time
    from blower_inspection.stationary_workers import CameraSnapshot
    view, _calls = window
    frame = np.full((80, 200, 3), 100, np.uint8)
    queued = [(1, StationaryCapture(60, frame[:40, :120], True, 100, burst_frames=5))]
    snapshot = CameraSnapshot(17, time.monotonic(), frame, "CAPTURING 120°", 30)
    acquisition = SimpleNamespace(snapshot=lambda: snapshot, take_captures=lambda: queued.copy())
    view.camera_worker = acquisition
    view.live_roi_bounds = (0, 0, 120, 40)
    view.part_detector = object()
    view.inference_worker = SimpleNamespace(track_id=1)
    def forbidden_read():
        raise AssertionError("The GUI must not read the camera owned by the acquisition worker")
    monkeypatch.setattr(view.camera, "read", forbidden_read)
    monkeypatch.setattr(view, "_locked_roi_tracks", lambda _frame: [TrackedPart(1, (0, 0, 120, 40), .99)])
    try:
        view._process_live_frame()
        assert view.camera_sequence == 17 and len(view.stationary_queue) == 1
        assert view.stationary_queue[0][1].burst_frames == 5
        view._tick()
        assert "30.0" in view.fps_top.text()
    finally:
        view.camera_worker = None
        view.inference_worker = None


def test_missing_patchcore_asset_is_reported_before_camera_open(window, monkeypatch):
    from dataclasses import replace
    view, _calls = window
    view.inspection_running = False
    model = replace(view.selected_model(), yolo_model_path=view.selected_model().model_file.with_name("yolo.pt"))
    messages = []
    monkeypatch.setattr(view, "selected_model", lambda: model)
    monkeypatch.setattr(view, "_apply_selected_camera_settings", lambda: None)
    monkeypatch.setattr(ui.QMessageBox, "critical", lambda _window, title, message: messages.append((title, message)))
    monkeypatch.setattr(view.camera, "open", lambda: pytest.fail("Camera opened before missing assets were reported"))
    view.start_inspection()
    assert messages[0][0] == "PatchCore model not ready"
    assert "PatchCore checkpoint not found" in messages[0][1]
    assert not view.inspection_running


def test_capture_packet_waits_for_first_snapshot_before_dispatch(window, monkeypatch):
    view, _calls = window
    frame = np.full((40, 120, 3), 100, np.uint8)
    packet = StationaryCapture(60, frame, True, 100, burst_frames=4)
    view.camera_worker = SimpleNamespace(snapshot=lambda: None, take_captures=lambda: [(1, packet)])
    dispatched = []
    monkeypatch.setattr(view, "_dispatch_stationary_view", lambda: dispatched.append(True))
    try:
        view._process_live_frame()
        assert not dispatched
        assert len(view.stationary_queue) == 1
    finally:
        view.camera_worker = None


def test_training_cannot_mutate_the_backend_during_inspection(window, monkeypatch):
    view, _calls = window
    messages = []
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *_args: messages.append(_args[1]))
    monkeypatch.setattr(ui, "TrainWorker", lambda *_args: pytest.fail("Training started during inspection"))
    view.train_selected()
    assert messages == ["Inspection active"]
    assert view.train_worker is None


def test_full_resolution_preview_is_resized_before_color_conversion(window, monkeypatch):
    view, _calls = window
    source = np.zeros((2160, 3840, 3), np.uint8)
    converted = []
    original = ui.cv2.cvtColor
    def conversion(frame, *args, **kwargs):
        converted.append(frame.shape)
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(ui.cv2, "cvtColor", conversion)
    view.show_frame(source)
    assert converted and max(shape[1] for shape in converted) <= view.viewer.width()
    assert max(shape[0] for shape in converted) <= view.viewer.height()
    assert view.preview_meter.count == 1
    assert source.shape == (2160, 3840, 3)


def test_separate_stage_metrics_use_existing_panel_and_do_not_change_layout(window):
    from blower_inspection.stationary_workers import CameraSnapshot
    from blower_inspection.capture_performance import StageMeter
    view, _calls = window
    image = np.zeros((40, 120, 3), np.uint8)
    meter = StageMeter().snapshot()
    performance = dict(raw={**meter, "fps": 30, "mean_ms": 33.3}, processing={**meter, "fps": 29},
                       quality=meter, motion=meter, evidence_dropped=0, preview_overwritten=7,
                       format=dict(width=3840, height=2160, fourcc="MJPG", backend="DSHOW"))
    view.camera_worker = SimpleNamespace(snapshot=lambda: CameraSnapshot(1, 0, image, "ROTATING", 30),
                                         performance=lambda: performance)
    view.inference_latency_ms = 75
    try:
        view._tick()
        text = view.last_result.text()
        for metric in ("RAW 30.0", "PROC 29.0", "GUI", "READ 33.3", "INFER 75.0", "DROP 0", "OVERWRITTEN 7", "3840x2160 MJPG DSHOW"):
            assert metric in text
        view._tick()
        assert view.last_result.text().count("\nRAW ") == 1
    finally:
        view.camera_worker = None


def test_evidence_queue_overflow_inhibits_inspection_without_dispatching_good_views(window, monkeypatch):
    view, calls = window
    image = np.zeros((40, 120, 3), np.uint8)
    packet = StationaryCapture(60, image, True, 100)
    for _ in range(view.stationary_queue.max_items):
        view.stationary_queue.append((1, packet))
    view.camera_worker = SimpleNamespace(snapshot=lambda: None, take_captures=lambda: [(1, packet)])
    monkeypatch.setattr(view, "stop_camera", lambda: setattr(view, "inspection_running", False))
    messages = []
    monkeypatch.setattr(ui.QMessageBox, "critical", lambda *_args: messages.append(_args[-1]))
    try:
        view._process_live_frame()
        assert calls == ["fail"] and view.status_badge.text() == "SYSTEM FAULT"
        assert "queue full" in messages[0]
        assert view.inference_worker is None
    finally:
        view.camera_worker = None


def test_unlocked_tracking_keeps_existing_stationary_capture_fallback(window, monkeypatch):
    from blower_inspection.stationary_workers import CameraSnapshot
    import time
    view, _calls = window
    source = np.full((80, 200, 3), 100, np.uint8)
    snapshot = CameraSnapshot(1, time.monotonic(), source, "STARTING CAMERA", 30)
    view.camera_worker = SimpleNamespace(snapshot=lambda: snapshot, take_captures=lambda: [])
    view.live_roi_bounds = None
    view.part_detector = SimpleNamespace(track=lambda _frame: [TrackedPart(1, (0, 0, 120, 40), .99)])
    offered = []
    monkeypatch.setattr(view, "_capture_stationary_views", lambda frame, _tracks: offered.append(frame))
    try:
        view._process_live_frame()
        assert len(offered) == 1 and offered[0] is source
    finally:
        view.camera_worker = None


def test_training_button_uses_a_separate_production_backend(window, monkeypatch):
    view, _calls = window
    view.inspection_running = False
    workers = []
    class Signal:
        def connect(self, _callback): pass
    class Worker:
        def __init__(self, inspector, model):
            self.inspector, self.model = inspector, model
            self.finished_ok, self.progress, self.failed = Signal(), Signal(), Signal()
            workers.append(self)
        def start(self): pass
    monkeypatch.setattr(ui, "TrainWorker", Worker)
    view.train_selected()
    assert len(workers) == 1
    assert type(workers[0].inspector).__name__ == "PatchCoreInspector"
    assert workers[0].inspector is not view.inspector
    assert workers[0].model == view.selected_model()
