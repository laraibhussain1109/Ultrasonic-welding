"""Qt interaction and threaded machine/capture integration tests."""
import json
import os
import queue
import socket
import threading
import time
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from blower_inspection.auth import User
from blower_inspection.fixed_settings import InspectionSettings
from blower_inspection.fixed_view_app import FixedInspectionWindow, STYLE
from blower_inspection.fixed_workers import SessionWorker
from blower_inspection.machine_signals import MachineEvent, TCPMachineConnection
from test_fixed_inspection import make_view


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(STYLE)
    yield application


def pump(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("Timed out waiting for threaded inspection event")


def test_gui_pages_live_heatmap_tuning_and_operator_permissions(app, tmp_path):
    path = tmp_path / "settings.json"
    InspectionSettings().save(path)
    window = FixedInspectionWindow(User("engineer", "admin"), path)
    window.show()
    app.processEvents()
    assert window.tabs.count() == 4
    assert window.read_settings() == window.settings
    view = make_view(failed=True)
    window.receive_view(view)
    window.tuning.inputs["heatmap_threshold"].setValue(1.1)
    assert "ENGINEERING PREVIEW: PASS" in window.engineering_stats.text()
    assert window.current_view.verdict == "FAIL"
    for mode in ["ORIGINAL", "ROI", "HEATMAP", "OVERLAY", "BINARY MASK", "FILTERED MASK", "FINAL REGIONS"]:
        window.display_mode.setCurrentText(mode)
        assert window.engineering_viewer.frame is not None
    window.close()
    operator = FixedInspectionWindow(User("operator", "user"), path)
    assert not operator.tabs.isTabEnabled(1)
    assert not operator.tabs.isTabEnabled(2)
    operator.close()


def test_calibration_recalculates_in_worker_without_model_calls(app, tmp_path):
    from blower_inspection.tolerance_calibration import CalibrationSample
    path = tmp_path / "settings.json"
    InspectionSettings().save(path)
    window = FixedInspectionWindow(User("engineer", "admin"), path)
    window.samples = [CalibrationSample("good", "GOOD", make_view()), CalibrationSample("ng", "NG", make_view(failed=True))]
    window.recalculate_calibration()
    pump(app, lambda: len(window.calibration_rows) == 2)
    assert "FNR: 0.00%" in window.calibration_summary.text()
    window.calibration_tuning.inputs["heatmap_threshold"].setValue(1.1)
    pump(app, lambda: "FNR: 100.00%" in window.calibration_summary.text())
    pump(app, lambda: not window.calibration_worker.isRunning())
    window.close()


def worker_fixture(monkeypatch, tmp_path, delay=0, fail_angle=300):
    defaults = InspectionSettings()
    settings = replace(defaults, camera=replace(defaults.camera, burst_frames=10),
        machine=replace(defaults.machine, interface="tcp_json", output_enabled=True,
                        settle_delay_s=0, acquisition_time_s=.2, minimum_burst_frames=2, position_timeout_s=1),
        storage=replace(defaults.storage, results_dir=str(tmp_path), save_fail_images=True))
    input_queue = queue.Queue()
    acknowledgments, output_calls, inferred = [], [], []
    class Camera:
        sequence = 0
        def frame_snapshot(self):
            self.sequence += 1
            return self.sequence, time.monotonic(), np.zeros((10,10,3), np.uint8)
    class Pipeline:
        def __init__(self, settings): pass
        def ready(self): pass
        def inspect_frames(self, frames, angle):
            assert len(frames) >= 2
            inferred.append(angle)
            if delay:
                time.sleep(delay)
            return make_view(angle, failed=angle==fail_angle)
    class Connection:
        def __init__(self, host, port): pass
        def poll(self):
            events = []
            while not input_queue.empty(): events.append(input_queue.get_nowait())
            return events
        def acknowledge(self, payload): acknowledgments.append(payload)
        def close(self): pass
    class Bridge:
        def __init__(self):
            from blower_inspection.fail_output import FailOutputConfig
            self.config = FailOutputConfig()
        def reset(self): output_calls.append("RESET")
        def send_result(self, failed): output_calls.append("FAIL" if failed else "CLEAR")
        def signal_pass(self): output_calls.append("PASS_PULSE")
        def close(self): pass
    monkeypatch.setattr("blower_inspection.fixed_workers.FixedViewPipeline", Pipeline)
    monkeypatch.setattr("blower_inspection.fixed_workers.TCPMachineConnection", Connection)
    monkeypatch.setattr("blower_inspection.fixed_workers.ESP32FailOutputBridge", Bridge)
    worker = SessionWorker(settings, Camera())
    return worker, input_queue, acknowledgments, output_calls, inferred


def test_threaded_plc_six_view_cycle_saves_every_side_and_never_pulses_pass_for_fail(app, tmp_path, monkeypatch):
    worker, events, acks, outputs, inferred = worker_fixture(monkeypatch, tmp_path)
    views, parts, errors = [], [], []
    worker.view_ready.connect(views.append)
    worker.part_ready.connect(parts.append)
    worker.failed.connect(errors.append)
    worker.start()
    events.put(MachineEvent("begin", "PART", fit_check_complete=True))
    try:
        for index, angle in enumerate(range(60,361,60)):
            if index: events.put(MachineEvent("moving"))
            events.put(MachineEvent("stopped", angle=angle))
            pump(app, lambda: len(views) == index+1)
        pump(app, lambda: len(parts) == 1)
        assert inferred == [60,120,180,240,300,360]
        assert parts[0]["final_verdict"] == "FAIL" and parts[0]["failed_views"] == [300]
        assert [ack["angle"] for ack in acks if ack["event"] == "view_result"] == inferred
        assert "PASS_PULSE" not in outputs and "FAIL" in outputs
        assert not errors
    finally:
        worker.requestInterruption()
        assert worker.wait(5000)
    report = next(tmp_path.glob("*/*/result.json"))
    assert json.loads(report.read_text())["view_count"] == 6


def test_motion_during_inference_discards_result_and_retries_same_angle(app, tmp_path, monkeypatch):
    worker, events, acks, outputs, inferred = worker_fixture(monkeypatch, tmp_path, delay=.15)
    views = []
    worker.view_ready.connect(views.append)
    worker.start()
    events.put(MachineEvent("begin", "PART", fit_check_complete=True))
    events.put(MachineEvent("stopped", angle=60))
    try:
        pump(app, lambda: inferred == [60])
        events.put(MachineEvent("moving"))
        time.sleep(.2)
        app.processEvents()
        assert not views and not acks
        events.put(MachineEvent("stopped", angle=60))
        pump(app, lambda: len(views) == 1)
        assert views[0].angle == 60 and inferred == [60,60]
    finally:
        worker.requestInterruption()
        assert worker.wait(5000)
    assert "PASS_PULSE" not in outputs


def test_pass_pulse_occurs_once_only_after_all_six_valid_views_are_saved(app, tmp_path, monkeypatch):
    worker, events, acks, outputs, inferred = worker_fixture(monkeypatch, tmp_path, fail_angle=None)
    views, parts, errors = [], [], []
    worker.view_ready.connect(views.append)
    worker.part_ready.connect(parts.append)
    worker.failed.connect(errors.append)
    worker.start()
    events.put(MachineEvent("begin", "PART", fit_check_complete=True))
    try:
        for index, angle in enumerate(range(60,361,60)):
            if index: events.put(MachineEvent("moving"))
            events.put(MachineEvent("stopped", angle=angle))
            pump(app, lambda: len(views) == index+1)
            if index < 5:
                assert "PASS_PULSE" not in outputs
        pump(app, lambda: len(parts)==1)
        events.put(MachineEvent("stopped", angle=360))
        events.put(MachineEvent("heartbeat"))
        app.processEvents()
        assert outputs.count("PASS_PULSE") == 1
        assert parts[0]["final_verdict"] == "PASS" and not errors
    finally:
        worker.requestInterruption()
        assert worker.wait(5000)
    report = json.loads(next(tmp_path.glob("*/*/result.json")).read_text())
    assert report["final_verdict"] == "PASS" and report["view_count"] == 6


def test_failed_evidence_save_faults_cycle_without_emitting_pass(app, tmp_path, monkeypatch):
    worker, events, acks, outputs, inferred = worker_fixture(monkeypatch, tmp_path)
    from blower_inspection.inspection_storage import ResultStorage
    def broken_save(self, view):
        raise OSError("disk full")
    monkeypatch.setattr(ResultStorage, "save_view", broken_save)
    errors = []
    worker.failed.connect(errors.append)
    worker.start()
    events.put(MachineEvent("begin", "PART", fit_check_complete=True))
    events.put(MachineEvent("stopped", angle=60))
    pump(app, lambda: not worker.isRunning())
    assert worker.wait(1000)
    assert errors and "disk full" in errors[0]
    assert not acks and "PASS_PULSE" not in outputs and "FAIL" in outputs
    assert json.loads(next(tmp_path.glob("*/*/result.json")).read_text())["final_verdict"] == "FAULT"


def test_lost_motor_state_heartbeat_faults_incomplete_part(app, tmp_path, monkeypatch):
    worker, events, acks, outputs, inferred = worker_fixture(monkeypatch, tmp_path)
    errors = []
    worker.failed.connect(errors.append)
    worker.start()
    events.put(MachineEvent("begin", "PART", fit_check_complete=True))
    pump(app, lambda: not worker.isRunning())
    assert worker.wait(1000)
    assert errors and "heartbeat" in errors[0]
    assert not inferred and "PASS_PULSE" not in outputs


def test_plc_protocol_reads_partial_lines_and_acknowledges_results():
    listener = socket.socket()
    listener.bind(("127.0.0.1",0))
    listener.listen()
    received = []
    def gateway():
        connection, _ = listener.accept()
        with connection:
            connection.sendall(b'{"event":"stop')
            time.sleep(.02)
            connection.sendall(b'ped","angle":60}\n{"event":"heartbeat"}\n')
            received.append(connection.recv(4096))
        listener.close()
    thread = threading.Thread(target=gateway)
    thread.start()
    bridge = TCPMachineConnection(*listener.getsockname())
    events = []
    try:
        deadline = time.monotonic()+2
        while len(events)<2 and time.monotonic()<deadline:
            events.extend(bridge.poll())
        assert [event.kind for event in events] == ["stopped", "heartbeat"]
        bridge.acknowledge({"event":"view_result", "angle":60, "verdict":"PASS"})
    finally:
        bridge.close()
        thread.join(2)
    assert json.loads(received[0])["angle"] == 60
