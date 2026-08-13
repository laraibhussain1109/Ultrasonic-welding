from blower_inspection.yolo_tracking import RotatingPartInspector, TrackedPart


def tracked(track_id, center_x, width=20):
    return TrackedPart(track_id, (center_x - width // 2, 20, width, 40), 0.9)


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
