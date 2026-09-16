import numpy as np

from blower_inspection.yolo_tracking import RotatingPartInspector, TrackedPart, YoloByteTrackDetector


def tracked(track_id, center_x, width=20):
    return TrackedPart(track_id, (center_x - width // 2, 20, width, 40), 0.9)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, item):
        return FakeTensor(self.value[item])


def test_startup_roi_selects_one_best_detection_without_bytetrack_id(monkeypatch, tmp_path):
    detector = YoloByteTrackDetector(tmp_path / "best.pt", confidence=.70)
    boxes = type("Boxes", (), {
        "conf": FakeTensor([.71, .94]),
        "xyxy": FakeTensor([[5, 10, 80, 50], [10, 20, 110, 70]]),
        "__len__": lambda self: 2,
    })()
    result = type("Result", (), {"boxes": boxes})()
    model = type("Model", (), {"predict": lambda self, *_args, **_kwargs: [result]})()
    monkeypatch.setattr(detector, "_load", lambda: model)

    selected = detector.detect_best(np.zeros((80, 120, 3), np.uint8))

    assert selected.bounds == (10, 20, 100, 50)
    assert selected.confidence == .94


def test_failed_surface_stays_latched_until_one_line_crossing_verdict():
    inspector = RotatingPartInspector(counting_line_ratio=0.8)
    inspector.observe_tracks([tracked(7, 40)], frame_width=100, now=10.0)
    assert inspector.record_inspection(7, is_pass=False, anomaly_score=0.9) is None
    inspector.observe_tracks([tracked(7, 55)], frame_width=100, now=10.2)
    assert inspector.record_inspection(7, is_pass=True, anomaly_score=0.2) is None
    assert inspector.latched_failure(7) is True

    inspector.observe_tracks([tracked(7, 85)], frame_width=100, now=10.4)
    completed = inspector.record_inspection(7, is_pass=True, anomaly_score=0.1)

    assert completed is not None
    assert completed.track_id == 7
    assert completed.status == "FAIL"
    assert completed.frames_inspected == 3
    assert completed.worst_score == 0.9


def test_disappearance_does_not_count_a_part_that_never_crossed_line():
    inspector = RotatingPartInspector(lost_timeout_s=0.5, counting_line_ratio=0.8)
    inspector.observe_tracks([tracked(1, 40)], frame_width=100, now=0.0)
    inspector.record_inspection(1, is_pass=True, anomaly_score=0.1)
    inspector.observe_tracks([], frame_width=100, now=1.0)

    assert inspector.flush() == []


def test_failure_survives_long_detection_gap_before_reappearing():
    inspector = RotatingPartInspector(lost_timeout_s=0.5, counting_line_ratio=0.8)
    inspector.observe_tracks([tracked(1, 40)], frame_width=100, now=0.0)
    inspector.record_inspection(1, is_pass=False, anomaly_score=0.8)
    inspector.observe_tracks([], frame_width=100, now=5.0)

    inspector.observe_tracks([tracked(2, 42)], frame_width=100, now=5.1)

    assert inspector.latched_failure(2) is True


def test_changed_bytetrack_id_is_reattached_to_same_rotating_part():
    inspector = RotatingPartInspector(lost_timeout_s=1.0, counting_line_ratio=0.8)
    inspector.observe_tracks([tracked(11, 45)], frame_width=100, now=0.0)
    inspector.record_inspection(11, is_pass=False, anomaly_score=0.7)
    inspector.observe_tracks([tracked(22, 47)], frame_width=100, now=0.2)
    assert inspector.latched_failure(22) is True
    inspector.record_inspection(22, is_pass=True, anomaly_score=0.1)
    inspector.observe_tracks([tracked(22, 85)], frame_width=100, now=0.4)
    completed = inspector.record_inspection(22, is_pass=True, anomaly_score=0.1)

    assert completed is not None
    assert completed.track_id == 11
    assert completed.status == "FAIL"


def test_part_first_seen_beyond_line_is_not_counted():
    inspector = RotatingPartInspector(counting_line_ratio=0.8)
    inspector.observe_tracks([tracked(5, 90)], frame_width=100, now=0.0)

    assert inspector.record_inspection(5, is_pass=True, anomaly_score=0.1) is None
    assert inspector.flush() == []


def test_crossed_part_waits_for_minimum_surface_view_count():
    inspector = RotatingPartInspector(
        counting_line_ratio=0.8, minimum_rotation_views=3
    )
    inspector.observe_tracks([tracked(3, 40)], frame_width=100, now=0.0)
    assert inspector.record_inspection(3, is_pass=True, anomaly_score=0.1) is None
    inspector.observe_tracks([tracked(3, 85)], frame_width=100, now=0.1)
    assert inspector.record_inspection(3, is_pass=True, anomaly_score=0.1) is None
    assert inspector.needs_completion_inspection(3) is True

    completed = inspector.record_inspection(3, is_pass=True, anomaly_score=0.1)

    assert completed is not None
    assert completed.frames_inspected == 3


def test_fixed_station_part_completes_after_minimum_rotation_views_once():
    inspector = RotatingPartInspector(
        counting_line_ratio=0.8, minimum_rotation_views=3,
        completion_mode="minimum_views",
    )
    inspector.observe_tracks([tracked(9, 50)], frame_width=100, now=0.0)

    assert inspector.record_inspection(9, is_pass=True, anomaly_score=.2) is None
    assert inspector.record_inspection(9, is_pass=True, anomaly_score=.1) is None
    completed = inspector.record_inspection(9, is_pass=True, anomaly_score=.1)

    assert completed is not None
    assert completed.status == "PASS"
    assert completed.valid_views == 3
    assert not inspector.accepts_inspection(9)

    # The same ByteTrack ID cannot be counted repeatedly while the part remains.
    inspector.observe_tracks([tracked(9, 50)], frame_width=100, now=1.0)
    assert not inspector.accepts_inspection(9)
    inspector.observe_tracks([], frame_width=100, now=2.0)
    inspector.observe_tracks([tracked(9, 50)], frame_width=100, now=3.0)
    assert inspector.accepts_inspection(9)


def test_fixed_station_fails_closed_after_too_many_invalid_views():
    inspector = RotatingPartInspector(minimum_rotation_views=2, completion_mode="minimum_views")
    inspector.observe_tracks([tracked(4, 50)], frame_width=100)

    for _ in range(3):
        assert inspector.record_inspection(4, is_pass=False, anomaly_score=0,
                                           view_valid=False, immediate_failure=False) is None
    completed = inspector.record_inspection(4, is_pass=False, anomaly_score=0,
                                             view_valid=False, immediate_failure=False)

    assert completed is not None
    assert completed.status == "FAIL"
    assert "INSUFFICIENT_VIEW_QUALITY" in completed.reason_codes


def test_horizontal_counting_line_uses_vertical_part_motion():
    inspector = RotatingPartInspector(
        counting_line_ratio=.5, counting_axis="y", counting_direction="top_to_bottom"
    )
    above = TrackedPart(12, (20, 20, 40, 20), .9)
    below = TrackedPart(12, (20, 60, 40, 20), .9)

    inspector.observe_tracks([above], frame_width=100, frame_height=100)
    inspector.record_inspection(12, is_pass=True, anomaly_score=.1)
    inspector.observe_tracks([below], frame_width=100, frame_height=100)
    completed = inspector.record_inspection(12, is_pass=True, anomaly_score=.1)

    assert completed is not None
    assert completed.status == "PASS"
