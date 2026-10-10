"""Camera-only recovery checks; physical commissioning is still required."""
import cv2
import numpy as np
import pytest

from blower_inspection.camera_modes import choose_mode, configure_capture
from blower_inspection.stationary_capture import INSPECTION_ANGLES, StationaryViewCapture
from blower_inspection.yolo_tracking import RotatingPartInspector, TrackedPart


def test_format_is_reasserted_after_size_and_fps_and_exposure_is_not_forced():
    class ResettingDriver:
        properties = {}
        def set(self, prop, value):
            self.properties[prop] = value
            if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS):
                self.properties[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*"YUY2")
            return True
    capture = ResettingDriver()
    accepted = configure_capture(capture, 3840, 2160, 30)
    assert capture.properties[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
    assert "auto_exposure" not in accepted and cv2.CAP_PROP_AUTO_EXPOSURE not in capture.properties
    configure_capture(capture, 3840, 2160, 30, auto_exposure=.75)
    assert capture.properties[cv2.CAP_PROP_AUTO_EXPOSURE] == .75


def report(width=3840, height=2160, fps=29, fourcc="MJPG"):
    return {"status": "ok", "frames": 30, "delivered_fps": fps,
            "actual": {"width": width, "height": height, "fourcc": fourcc, "reported_fps": 30}}


def test_choose_mode_uses_reads_and_native_size_instead_of_reported_30fps(monkeypatch):
    monkeypatch.delenv("NEUROIRIS_CAMERA_BACKEND", raising=False)
    monkeypatch.delenv("NEUROIRIS_CAMERA_FOURCC", raising=False)
    calls = []
    def probe(_index, backend, _width, _height, fourcc, _fps, _exposure):
        calls.append((backend, fourcc))
        return report(fps=2, fourcc="YUY2") if backend == "DSHOW" else report()
    selected, reports = choose_mode(0, 3840, 2160, 30, probe=probe)
    assert selected["probe_backend"] == "MSMF" and selected["delivered_fps"] == 29
    assert calls == [("DSHOW", "MJPG"), ("MSMF", "MJPG")] and len(reports) == 2
    with pytest.raises(RuntimeError, match="resolution was not reduced"):
        choose_mode(0, 3840, 2160, 30, probe=lambda *_args: report(width=1920, height=1080, fps=60))


def test_camera_mode_overrides_are_honored_and_slow_actual_rate_remains_visible(monkeypatch):
    monkeypatch.setenv("NEUROIRIS_CAMERA_BACKEND", "DSHOW")
    monkeypatch.setenv("NEUROIRIS_CAMERA_FOURCC", "YUY2")
    calls = []
    def probe(*args):
        calls.append(args)
        return report(fps=2, fourcc="YUY2")
    selected, _ = choose_mode(0, 3840, 2160, 30, probe=probe)
    assert selected["delivered_fps"] == 2 and len(calls) == 1
    assert calls[0][1] == "DSHOW" and calls[0][4] == "YUY2"


def test_retained_manual_exposure_is_compared_with_backend_automatic_exposure(monkeypatch):
    monkeypatch.delenv("NEUROIRIS_CAMERA_BACKEND", raising=False)
    monkeypatch.delenv("NEUROIRIS_CAMERA_FOURCC", raising=False)
    calls = []
    def probe(_index, backend, _width, _height, fourcc, _fps, exposure):
        calls.append((backend, fourcc, exposure))
        return report(fps=29 if exposure == .75 else 2)
    selected, _ = choose_mode(0, 3840, 2160, 30, probe=probe)
    assert selected["probe_backend"] == "DSHOW" and selected["probe_auto_exposure"] == .75
    assert selected["delivered_fps"] == 29 and len(calls) == 5


def test_camera_read_rejects_a_driver_resolution_fallback(monkeypatch):
    from blower_inspection.camera import USBCamera
    class Capture:
        def __init__(self, *_args): pass
        def isOpened(self): return True
        def set(self, *_args): return True
        def get(self, _prop): return 0
        def getBackendName(self): return "DSHOW"
        def read(self): return True, np.zeros((1080, 1920, 3), np.uint8)
        def release(self): pass
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setenv("NEUROIRIS_CAMERA_AUTOSELECT", "0")
    camera = USBCamera(width=3840, height=2160)
    try:
        with pytest.raises(RuntimeError, match="inspection inhibited"):
            camera.read()
    finally:
        camera.close()


def subtle_frames():
    background = np.full((100, 600, 3), 100, np.uint8)
    texture = np.random.default_rng(44).integers(30, 220, (40, 70, 3), dtype=np.uint8)
    frames = []
    for shift in range(12):
        frame = background.copy()
        frame[30:70, 260:330] = np.roll(texture, shift, axis=1)
        frames.append(frame)
    return frames


def test_localized_subtle_steps_below_both_old_global_thresholds_produce_six_stops():
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        frames = subtle_frames()
        capture = StationaryViewCapture(burst_frames=15, settle_ms=200)
        # Demonstrate why the previous global motion gate missed this sequence.
        for previous, current in zip(frames, frames[1:]):
            before, after = capture._motion_image(previous), capture._motion_image(current)
            delta = after.astype(np.float32) - before.astype(np.float32)
            flow = cv2.calcOpticalFlowFarneback(before, after, None, .5, 3, 15, 2, 5, 1.2, 0)
            assert np.mean(np.abs(delta - np.median(delta))) < capture.motion_threshold
            assert np.quantile(cv2.magnitude(flow[..., 0], flow[..., 1]), .80) < capture.flow_threshold
        packets, index = [], 0
        for _stop in range(7):  # fitted/home, followed by six mechanical steps
            for frame in frames + [frames[-1]] * 30:
                selected = capture.offer(frame, now=index / 30)
                index += 1
                if selected:
                    packets.append(selected)
        assert [packet.angle for packet in packets] == list(INSPECTION_ANGLES)
        assert all(packet.valid and packet.burst_frames >= 3 for packet in packets)
        assert all(packet.frame.shape == (100, 600, 3) for packet in packets)
    finally:
        cv2.setNumThreads(old_threads)


def test_stationary_sensor_noise_and_brightness_changes_do_not_invent_steps():
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        rng = np.random.default_rng(112)
        base = rng.integers(50, 180, (100, 600, 3), dtype=np.uint8)
        capture = StationaryViewCapture(skip_fit_rotation=False)
        for index in range(80):
            noise = rng.integers(-1, 2, base.shape)
            frame = np.clip(base.astype(np.int16) + noise + (index % 10) * 3, 0, 255).astype(np.uint8)
            assert capture.offer(frame, now=index / 30) is None
        assert capture._next_index == 0
    finally:
        cv2.setNumThreads(old_threads)


def test_repeated_video_passes_and_candidates_never_complete_or_inflate_coverage():
    inspector = RotatingPartInspector(completion_mode="minimum_views", fixed_view_angles=INSPECTION_ANGLES)
    inspector.observe_tracks([TrackedPart(1, (0, 0, 600, 100), .99)], 700)
    for _frame in range(60):
        inspector.record_video_evidence(1, view_valid=True, immediate_failure=False,
                                        anomaly_score=2, candidate_sections=(2,))
    assert inspector.view_progress(1) == (0, 0, 6)
    assert inspector.accepts_inspection(1) and not inspector.latched_failure(1)
    completed = inspector.flush()[0]
    assert completed.status == "FAIL" and completed.valid_views == 0


def test_valid_video_defects_latch_sections_without_assigning_an_unknown_angle():
    inspector = RotatingPartInspector(completion_mode="minimum_views", fixed_view_angles=INSPECTION_ANGLES)
    inspector.observe_tracks([TrackedPart(1, (0, 0, 600, 100), .99)], 700)
    inspector.record_video_evidence(1, view_valid=False, immediate_failure=True, anomaly_score=3,
                                    candidate_sections=(2,))
    assert not inspector.latched_failure(1)
    inspector.record_video_evidence(1, view_valid=True, immediate_failure=True, anomaly_score=3,
                                    candidate_sections=(2,), reason_codes=("CONFIRMED_DEFECT",))
    assert inspector.latched_failure(1) and inspector.defect_sections(1) == (2,)
    assert inspector.view_progress(1) == (0, 0, 6) and inspector.defect_section_angles(1) == {}
    for angle in INSPECTION_ANGLES:
        completed = inspector.record_inspection(1, is_pass=True, anomaly_score=.2, view_angle=angle)
    assert completed.status == "FAIL" and completed.valid_views == 6
